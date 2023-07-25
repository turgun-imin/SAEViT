import numpy as np


class evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2, dtype=np.int64)
        self.eps = 1e-8

    def get_tp_fp_tn_fn(self):
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        tn = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)
        return tp, fp, tn, fn

    def overall_accuracy(self):
        oa = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)
        return oa

    def class_pixel_accuracy(self):
        acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) + self.eps)
        return acc

    def mean_pixel_accuracy(self):
        classacc = self.class_pixel_accuracy()
        macc = np.nanmean(classacc)
        return macc

    def precision(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp)
        return precision

    def recall(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn)
        return recall

    def f1(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2.0 * precision * recall) / (precision + recall)
        return f1

    def dice_score(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        dice = 2 * tp / ((tp + fp) + (tp + fn))
        return dice

    def iou(self):
        iou = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(1) +
                                                self.confusion_matrix.sum(0) - np.diag(self.confusion_matrix))
        return iou

    def mean_iou(self):
        iou = self.iou()
        miou = np.nanmean(iou)
        return miou

    def fw_iou(self):
        freq = np.sum(self.confusion_matrix, axis=1) / (np.sum(self.confusion_matrix) + self.eps)
        iou = self.iou()
        fwiou = (freq[freq > 0] * iou[freq > 0]).sum()
        return fwiou

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype(int) + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, 'pre_image shape {}, gt_image shape {}'.format(pre_image.shape,
                                                                                                 gt_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


if __name__ == '__main__':

    gt = np.array([[0, 1, 1],
                   [0, 0, 1],
                   [1, 0, 1]])

    pre = np.array([[0, 1, 1],
                   [0, 0, 1],
                   [1, 1, 1]])

    eval = evaluator(num_class=2)
    eval.add_batch(gt, pre)
    print(eval.confusion_matrix)
    print(eval.get_tp_fp_tn_fn())
    print(eval.precision())
    print(eval.recall())
    print(eval.iou())
    print(eval.overall_accuracy())
    print(eval.f1())
    print(eval.fw_iou())
    print(eval.dice_score())
