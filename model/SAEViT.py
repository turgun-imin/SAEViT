from typing import Type, Callable, Tuple, Optional, Set, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from ptflops import get_model_complexity_info


import sys
import logging
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.SoftPool import SoftPool2d
from utils.DeformConv import DeformConv2d


class LayerNorm2d(nn.LayerNorm):
    def forward(self,
                input: torch.Tensor
    ) -> None:
        output = input.permute(0, 2, 3, 1)
        output = F.layer_norm(output, self.normalized_shape, self.weight, self.bias, self.eps)
        output = output.permute(0, 3, 1, 2)
        return output

class Conv(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (1, 1),
                 dilation: int = 1,
                 bias: bool = False,
    ) -> None:
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=padding))

class ConvBNReLU(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (1, 1),
                 dilation: int = 1,
                 bias: bool = False,
                 inplace: bool = True
    ) -> None:
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=inplace)
        )


class DepthWiseConv(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (1, 1),
                 dilation: int = 1,
                 bias: bool = False
    ) -> None:
        super(DepthWiseConv, self).__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels,
                      stride=stride, padding=padding, dilation=dilation, bias=bias),
        )

class DepthWiseSeparableConv(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dw_kernel_size: Tuple[int, int] = (3, 3),
                 pw_kernel_size: Tuple[int, int] = (1, 1),
                 dw_stride: Tuple[int, int] = (1, 1),
                 dw_padding: Tuple[int, int] = (1, 1),
                 pw_stride: Tuple[int, int] = (1, 1),
                 pw_padding: Tuple[int, int] = (0, 0),
                 dilation: int = 1,
                 bias: bool = False
    ) -> None:
        super(DepthWiseSeparableConv, self).__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=dw_kernel_size, groups=in_channels,
                      stride=dw_stride, padding=dw_padding, dilation=dilation, bias=bias),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=pw_kernel_size, groups=1,
                      stride=pw_stride, padding=pw_padding, dilation=dilation, bias=bias))


class DepthWiseSeparableConvBNReLU(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dw_kernel_size: Tuple[int, int] = (3, 3),
                 pw_kernel_size: Tuple[int, int] = (1, 1),
                 dw_stride: Tuple[int, int] = (1, 1),
                 dw_padding: Tuple[int, int] = (1, 1),
                 pw_stride: Tuple[int, int] = (1, 1),
                 pw_padding: Tuple[int, int] = (0, 0),
                 dilation: int = 1,
                 bias: bool = False
    ) -> None:
        super(DepthWiseSeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=dw_kernel_size, groups=in_channels,
                      stride=dw_stride, padding=dw_padding, dilation=dilation, bias=bias),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=pw_kernel_size, groups=1,
                      stride=pw_stride, padding=pw_padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class DepthWiseSeparableConvBNGELU(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dw_kernel_size: Tuple[int, int] = (3, 3),
                 pw_kernel_size: Tuple[int, int] = (1, 1),
                 dw_stride: Tuple[int, int] = (1, 1),
                 dw_padding: Tuple[int, int] = (1, 1),
                 pw_stride: Tuple[int, int] = (1, 1),
                 pw_padding: Tuple[int, int] = (0, 0),
                 dilation: int = 1,
                 bias: bool = False
    ) -> None:
        super(DepthWiseSeparableConvBNGELU, self).__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=dw_kernel_size, groups=in_channels,
                      stride=dw_stride, padding=dw_padding, dilation=dilation, bias=bias),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=pw_kernel_size, groups=1,
                      stride=pw_stride, padding=pw_padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )


class Patch_embedding(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 64,
    ) -> None:
        super(Patch_embedding, self).__init__()
        self.conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels // 2,
                                kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = ConvBNReLU(in_channels=out_channels // 2, out_channels=out_channels,
                                kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.skip_conv = ConvBNReLU(in_channels=out_channels, out_channels=out_channels,
                                kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv1(input)
        output = self.conv2(output)
        output = output + self.skip_conv(output)
        return output  # 8, 64, 256, 256


class Patch_merging(nn.Module):
    def __init__(self,
                 in_channels: int = 64,
                 out_channels: int = 96
    ) -> None:
        super(Patch_merging, self).__init__()
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            DepthWiseSeparableConvBNGELU(in_channels=in_channels, out_channels=out_channels,
                                         dw_kernel_size=(3, 3), dw_stride=(2, 2), dw_padding=(1, 1),
                                         pw_kernel_size=(1, 1), pw_stride=(1, 1), pw_padding=(0, 0)),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )
        self.pool_path = nn.Sequential(
            SoftPool2d(kernel_size=2, stride=2, adaptive=False),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv_path(input)
        output = output + self.pool_path(input)
        return output  # 8, 96, 128, 128


def window_partition(
        input: torch.Tensor,
        window_size: Tuple[int, int] = (8, 8)
) -> torch.Tensor:
    B, C, H, W = input.shape
    windows = input.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    windows = windows.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(
        windows: torch.Tensor,
        original_size: Tuple[int, int],
        window_size: Tuple[int, int] = (8, 8)
) -> torch.Tensor:
    H, W = original_size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    output = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    output = output.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return output


def interlaced_windows_partition(
        input: torch.Tensor,
        windows_size: Tuple[int, int] = (8, 8)
) -> torch.Tensor:
    B, C, H, W = input.shape
    grid = input.view(B, C, windows_size[0], H // windows_size[0], windows_size[1], W // windows_size[1])
    grid = grid.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, windows_size[0], windows_size[1], C)
    return grid


def interlaced_windows_reverse(
        windows: torch.Tensor,
        original_size: Tuple[int, int],
        windows_size: Tuple[int, int] = (8, 8)
) -> torch.Tensor:
    (H, W), C = original_size, windows.shape[-1]
    B = int(windows.shape[0] / (H * W / windows_size[0] / windows_size[1]))
    output = windows.view(B, H // windows_size[0], W // windows_size[1], windows_size[0], windows_size[1], C)
    output = output.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, C, H, W)
    return output


class self_attention(nn.Module):
    """ windows based multihead self-attention """
    def __init__(self,
                 in_channels: int,
                 num_heads: int,
                 window_size: Tuple[int, int],
                 partition_function: Callable,
                 reverse_function: Callable,
                 drop: float = 0.
    ) -> None:
        super(self_attention, self).__init__()

        self.in_channels: int = in_channels
        self.num_heads: int = num_heads
        self.window_size: Tuple[int, int] = window_size
        self.partition_function: Callable = partition_function
        self.reverse_function: Callable = reverse_function
        self.scale: float = num_heads ** -0.5

        self.qkv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 3, kernel_size=(1, 1), bias=False)
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1), bias=False)
        self.proj_drop = nn.Dropout(p=drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(input)
        B, C, H, W = qkv.shape
        qkv_partitioned = self.partition_function(qkv, self.window_size)
        qkv_partitioned = qkv_partitioned.view(-1, self.window_size[0] * self.window_size[1], C)
        b, n, c = qkv_partitioned.shape
        num_heads_qkv = qkv_partitioned.reshape(b, n, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = num_heads_qkv.unbind(0)
        q = q * self.scale
        attn = self.softmax(q @ k.transpose(-2, -1))
        output = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        output = self.reverse_function(output, (H, W), self.window_size)
        output = self.proj(output)
        output = self.proj_drop(output)
        return output


class CFFN(nn.Module):
    """  feed-forward network with convolution """
    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 act_layer: Type[nn.Module] = nn.GELU,
                 drop: float = 0.
    ) -> None:
        super(CFFN, self).__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_channels=in_features, out_channels=hidden_features, kernel_size=(1, 1))
        self.dwconv = DepthWiseConv(in_channels=hidden_features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.act_layer = act_layer()
        self.fc2 = nn.Conv2d(in_channels=hidden_features, out_channels=out_features, kernel_size=(1, 1))
        self.drop = nn.Dropout(drop)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.fc1(input)
        output = self.act_layer(output + self.dwconv(output))
        output = self.fc2(output)
        output = self.drop(output)
        return output


class SAEViT_block(nn.Module):
    """ SAEViT block """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_heads: int,
                 window_size: Tuple[int, int] = (8, 8),
                 drop: float = 0.,
                 drop_path: float = 0.,
                 mlp_ratio: float = 4.,
                 act_layer: Type[nn.Module] = nn.GELU,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 attention_norm_layer: Type[nn.Module] = LayerNorm2d
    ) -> None:
        super(SAEViT_block, self).__init__()

        self.patch_merging = nn.Sequential(
            norm_layer(in_channels),
            Patch_merging(
                in_channels=in_channels,
                out_channels=out_channels
            ))

        """ Interlaced windows partition multihead self-attention with convolution """
        self.WCMS = nn.Sequential(
            attention_norm_layer(out_channels),
            self_attention(
                in_channels=out_channels,
                num_heads=num_heads,
                window_size=window_size,
                partition_function=window_partition,
                reverse_function=window_reverse,
                drop=drop
            ),
            DropPath(drop_path) if drop_path > 0. else nn.Identity()
        )

        self.cffn_1 = nn.Sequential(
            attention_norm_layer(out_channels),
            CFFN(
                in_features=out_channels,
                hidden_features=int(out_channels * mlp_ratio),
                out_features=out_channels,
                act_layer=act_layer,
                drop=drop
            ),
            DropPath(drop_path) if drop_path > 0. else nn.Identity()
        )

        """ Windows partition multihead self-attention with convolution """
        self.IWCMS = nn.Sequential(
            attention_norm_layer(out_channels),
            self_attention(
                in_channels=out_channels,
                num_heads=num_heads,
                window_size=window_size,
                partition_function=interlaced_windows_partition,
                reverse_function=interlaced_windows_reverse,
                drop=drop
            ),
            DropPath(drop_path) if drop_path > 0. else nn.Identity()
        )

        self.cffn_2 = nn.Sequential(
            attention_norm_layer(out_channels),
            CFFN(
                in_features=out_channels,
                hidden_features=int(out_channels * mlp_ratio),
                out_features=out_channels,
                act_layer=act_layer,
                drop=drop
            ),
            DropPath(drop_path) if drop_path > 0. else nn.Identity()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.patch_merging(input)
        output = output + self.WCMS(output)
        output = output + self.cffn_1(output)
        output = output + self.IWCMS(output)
        output = output + self.cffn_2(output)
        return output


class SAEViT_stage(nn.Module):
    def __init__(self,
                 depth: int,
                 in_channels: int,
                 out_channels: int,
                 num_heads: int = 4,
                 window_size: Tuple[int, int] = (8, 8),
                 drop: float = 0.,
                 drop_path: Union[List[float], float] = 0.,
                 mlp_ratio: float = 4.,
                 act_layer: Type[nn.Module] = nn.GELU,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 attention_norm_layer: Type[nn.Module] = LayerNorm2d
    ) -> None:
        super(SAEViT_stage, self).__init__()

        self.blocks = nn.ModuleList([])
        for index in range(depth):
            self.blocks.append(
                SAEViT_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_heads=num_heads,
                    window_size=window_size,
                    drop=drop,
                    drop_path=drop_path,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    attention_norm_layer=attention_norm_layer
                ))

    def forward(self, input=torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            output = block(input)
        return output


class SAEViT_encoder(nn.Module):
    def __init__(self,
                 embed_dim: int = 64,
                 channels: Tuple[int, ...] = (96, 128, 256, 512),
                 depths: Tuple[int, ...] = (2, 2, 5, 2),
                 window_size: Tuple[int, int] = (8, 8),
                 num_heads: Tuple[int, ...] = (4, 8, 16, 32),
                 mlp_ratio: float = 4.,
                 drop: float = 0.,
                 drop_path_rate: float = 0.3,
                 act_layer: Type[nn.Module] = nn.GELU,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 attention_norm_layer: Type[nn.Module] = LayerNorm2d
    ) -> None:
        super(SAEViT_encoder, self).__init__()

        drop_path = [x.item() for x in torch.linspace(0, drop_path_rate, len(depths))]
        self.stage1 = SAEViT_stage(
            depth=depths[0],
            in_channels=embed_dim,
            out_channels=channels[0],
            num_heads=num_heads[0],
            window_size=window_size,
            drop=drop,
            drop_path=drop_path[0],
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
            attention_norm_layer=attention_norm_layer
        )
        self.stage2 = SAEViT_stage(
            depth=depths[1],
            in_channels=channels[0],
            out_channels=channels[1],
            num_heads=num_heads[1],
            window_size=window_size,
            drop=drop,
            drop_path=drop_path[1],
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
            attention_norm_layer=attention_norm_layer
        )
        self.stage3 = SAEViT_stage(
            depth=depths[2],
            in_channels=channels[1],
            out_channels=channels[2],
            num_heads=num_heads[2],
            window_size=window_size,
            drop=drop,
            drop_path=drop_path[2],
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
            attention_norm_layer=attention_norm_layer
        )
        self.stage4 = SAEViT_stage(
            depth=depths[3],
            in_channels=channels[2],
            out_channels=channels[3],
            num_heads=num_heads[3],
            window_size=window_size,
            drop=drop,
            drop_path=drop_path[3],
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
            attention_norm_layer=attention_norm_layer
        )

    def forward(self, input: torch.Tensor) -> List:
        features = []
        output = self.stage1(input)
        features.append(output)
        output = self.stage2(output)
        features.append(output)
        output = self.stage3(output)
        features.append(output)
        output = self.stage4(output)
        features.append(output)
        return features


if __name__ == '__main__':

    input = torch.rand(2, 64, 256, 256)

    model = SAEViT_encoder(
        embed_dim=64,
        channels=(96, 128, 256, 512),
        depths=(2, 2, 5, 2),
        window_size=(8, 8),
        num_heads=(4, 8, 16, 32),
    )
    outputs = model(input)
    print(outputs[0].shape)
    print(outputs[1].shape)
    print(outputs[2].shape)
    print(outputs[3].shape)


class SAEM_block(nn.Module):
    """ Shape-aware enhancement module block """
    def __init__(self,
                 dwun_sample: bool = True,
                 in_channels: int = 64,
                 out_channels: int = 96
    ) -> None:
        super(SAEM_block, self).__init__()

        """ downsampling part """
        self.ds = dwun_sample
        self.down_sample = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        )
        """ multi-shape kernels convolution part """
        self.block = nn.Sequential(
            
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.ds:
            output = self.down_sample(input)
            output = output + self.block(output)
        else:
            output = input + self.block(input)
        return output


class SAEM(nn.Module):
    """ Shape-aware enhancement Module """
    def __init__(self,
                 embed_dim: int = 64,
                 channels: Tuple[int, ...] = (96, 128, 256, 512),
    ) -> None:
        super(SAEM, self).__init__()

        self.saem_stage1 = SAEM_block(dwun_sample=True, in_channels=embed_dim, out_channels=channels[0])
        self.saem_stage2 = SAEM_block(dwun_sample=True, in_channels=channels[0], out_channels=channels[1])
        self.saem_stage3 = nn.Sequential(
            SAEM_block(dwun_sample=True, in_channels=channels[1], out_channels=channels[2]),
            SAEM_block(dwun_sample=False, in_channels=channels[2], out_channels=channels[2]),
        )
        self.saem_stage4 = SAEM_block(dwun_sample=True, in_channels=channels[2], out_channels=channels[3])

    def forward(self, input: torch.Tensor) -> List:
        features = []
        output = self.saem_stage1(input)
        features.append(output)
        output = self.saem_stage2(output)
        features.append(output)
        output = self.saem_stage3(output)
        features.append(output)
        output = self.saem_stage4(output)
        features.append(output)
        return features


# if __name__ == '__main__':

#     input = torch.rand(1, 64, 32, 32)

#     model = SAEM()
#     outputs = model(input)
#     print(outputs[0].shape)
#     print(outputs[1].shape)
#     print(outputs[2].shape)
#     print(outputs[3].shape)


class MPCA_block(nn.Module):
    """ Multi-pooling channel attention block """
    def __init__(self,
                 kernel_size: int = 3,
    ) -> None:
        super(MPCA_block, self).__init__()
        

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        return self.sigmoid(weightpool)


class MPCA(nn.Module):
    """ Multi-pooling channel attention """
    def __init__(self,
                 in_channels: int = 3,
                 embed_dim: int = 64,
                 channels: Tuple[int, ...] = (96, 128, 256, 512),
                 depths: Tuple[int, ...] = (2, 2, 5, 2),
                 window_size: Tuple[int, int] = (8, 8),
                 num_heads: Tuple[int, ...] = (4, 8, 16, 32)
    ) -> None:
        super(MPCA, self).__init__()

        self.patch_embedding = Patch_embedding(
            in_channels=in_channels,
            out_channels=embed_dim
        )
        self.transformer_encoder = SAEViT_encoder(
            embed_dim=embed_dim,
            channels=channels,
            depths=depths,
            window_size=window_size,
            num_heads=num_heads
        )
        self.mpca = MPCA_block()
        self.saem = SAEM(
            embed_dim=embed_dim,
            channels=channels
        )

    def forward(self, input: torch.Tensor) -> List:
        features = []
        pm = self.patch_embedding(input)
        t1, t2, t3, t4 = self.transformer_encoder(pm)
        c1, c2, c3, c4 = self.saem(pm)

        return features


# if __name__ == '__main__':

#     model = MPCA(
#         in_channels=3,
#         embed_dim=64,
#         channels=(96, 128, 256, 512),
#         depths=(2, 2, 5, 2),
#         window_size=(4, 4),
#         num_heads=(4, 8, 16, 32)
#     )
#     # macs, params = get_model_complexity_info(model, (3, 512, 512), print_per_layer_stat=False)
#     # print(f'\nComputational complexity: {macs:<8}\n'
#     #       f'Number of parameters: {params:<8}\n'
#     #       )

#     input = torch.rand(2, 3, 128, 128)
#     outputs = model(input)
#     print(outputs[0].shape)
#     print(outputs[1].shape)
#     print(outputs[2].shape)
#     print(outputs[3].shape)


class PAUM(nn.Module):
    """ Progressive aggregation upsampling module """
    def __init__(self,
                 num_classes: int = 2,
                 in_channels: int = 3,
                 embed_dim: int = 64,
                 channels: Tuple[int, ...] = (96, 128, 256, 512),
                 depths: Tuple[int, ...] = (2, 2, 5, 2),
                 window_size: Tuple[int, int] = (8, 8),
                 num_heads: Tuple[int, ...] = (4, 8, 16, 32)
    ) -> None:
        super(PAUM, self).__init__()

        self.backbone = MPCA(
            in_channels=in_channels,
            embed_dim=embed_dim,
            channels=channels,
            depths=depths,
            window_size=window_size,
            num_heads=num_heads
        )
        self.conv3 = nn.Sequential(

        )
        self.conv2 = nn.Sequential(

        )
        self.conv1 = nn.Sequential(

        )
        self.final_conv = nn.Sequential(

        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        f0, f1, f2, f3 = self.backbone(input)

        up_3_conv = f2 + self.conv3(up_3_att)

        up_2_conv = f1 + self.conv2(up_2_att)

        up_1_conv = f0 + self.conv1(up_1_att)

        final_up = F.interpolate(up_1_conv, input.size()[-2:], mode='bilinear', align_corners=False)
        output = self.final_conv(final_up)

        return output

def SAEViT():
    return PAUM(
        num_classes=2,
        in_channels=3,
        embed_dim=64,
        channels=(96, 128, 256, 512),
        depths=(2, 2, 5, 2),
        window_size=(8, 8),
        num_heads=(4, 8, 16, 32)
    )


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAEViT().to(device)
    # macs, params = get_model_complexity_info(model, (3, 512, 512), print_per_layer_stat=False)
    # print(f'\nComputational complexity: {macs:<8}\n'
    #       f'Number of parameters: {params:<8}\n'
    #       )

    # input = torch.rand(8, 3, 512, 512)
    input = torch.rand(1, 3, 256, 256).to(device)
    outputs = model(input)
    print(outputs.shape)

    # print(outputs[1].shape)
    # print(outputs[2].shape)
    # print(outputs[3].shape)
