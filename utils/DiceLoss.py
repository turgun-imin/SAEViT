import torch
from torch import nn

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1.0 - loss
        return loss

    def forward(self, inputs, label, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        if inputs.shape[1] != label.shape[1]:
            target = self._one_hot_encoder(label)
        else:
            target = label
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        dice_loss = 0.0
        if inputs.shape[1] == label.shape[1]:
            dice = self._dice_loss(inputs, target)
            dice_loss += dice
            # dice_loss = dice_loss / self.n_classes
        else:
            for i in range(0, self.n_classes):
                dice = self._dice_loss(inputs[:, i], target[:, i])
                dice_loss += dice * weight[i]
            # dice_loss = dice_loss / self.n_classes  # if any weight is 0 --> loss = loss

        return dice_loss