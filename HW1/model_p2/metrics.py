import numpy as np


def mean_iou_score(pred, labels):
    """
    Compute mean IoU score over 6 classes
    """
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        print('class #%d : %1.5f' % (i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou


class IOU(object):
    def __init__(self):
        self.unions = np.zeros(6)
        self.tps = np.zeros(6)
        self.iou = np.zeros(6)
        self.mean_iou = 0

    def batch_iou(self, pred, labels):
        """
        Compute mean IoU score over 6 classes for a batch and update
        IOU = overlap / union
        """
        for i in range(6):
            tp_fp = np.sum(pred == i)
            tp_fn = np.sum(labels == i)
            tp = np.sum((pred == i) * (labels == i))

            self.tps[i] += tp
            self.unions[i] += (tp_fp + tp_fn - tp)

    # def update(self):
    #     for i in range(6):
    #         if self.unions[i] != 0:
    #             iou = self.tps[i] / self.unions[i]
    #             self.mean_iou += iou/6

    def miou(self):
        mean_iou = 0
        for i in range(6):
            if self.unions[i] != 0:
                iou = self.tps[i] / self.unions[i]
                mean_iou += iou/6
        return mean_iou
