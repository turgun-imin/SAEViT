import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftPool2d(nn.Module):
    def __init__(self, kernel_size, stride, adaptive):
        super(SoftPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.adaptive = adaptive

    def forward(self, x):
        x = self.soft_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        return x

    def soft_pool2d(self, x, kernel_size=2, stride=None):

        kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        else:
            stride = (stride, stride)

        if self.adaptive:
            B, C, H, W = x.shape
            kernel_size = (H, W)
            stride = (H, W)

        _, c, h, w = x.shape
        e_x = torch.sum(torch.exp(x), dim=1, keepdim=True)
        return F.avg_pool2d(x * e_x, kernel_size, stride=stride) * (sum(kernel_size))/(F.avg_pool2d(e_x, kernel_size, stride=stride) * (sum(kernel_size)))


if __name__ == '__main__':
    input = torch.rand(8, 128, 256, 256)
    softpool = SoftPool2d(kernel_size=None, stride=None, adaptive=True)
    outputs = softpool(input)
    print(outputs.shape)
