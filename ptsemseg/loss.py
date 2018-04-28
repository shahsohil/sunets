import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix


class cross_entropy2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore=-100):
        super(cross_entropy2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight=weight, size_average=size_average, ignore_index=ignore)
        self.ignore = ignore

    def forward(self, input, target, th=1.0):
        log_p = F.log_softmax(input)
        if th < 1: # This is done while using Hardmining. Not use for our model training
            mask = F.softmax(input, dim=1) > th
            mask = mask.data
            new_target = target.data.clone()
            new_target[new_target == self.ignore] = 0
            indx = torch.gather(mask, 1, new_target.unsqueeze(1))
            indx = indx.squeeze(1)
            mod_target = target.clone()
            mod_target[indx] = self.ignore

        if th < 1:
            loss = self.nll_loss(log_p, mod_target)
            total_valid_pixel = torch.sum(mod_target.data != self.ignore)
        else:
            loss = self.nll_loss(log_p, target)
            total_valid_pixel = torch.sum(target.data != self.ignore)
        return loss, Variable(torch.FloatTensor([total_valid_pixel]).cuda())


def pixel_accuracy(outputs, labels, n_classes):
    lbl = labels.data
    mask = lbl < n_classes

    accuracy = []
    for output in outputs:
        _, pred = output.data.max(dim=1)
        diff = pred[mask] - lbl[mask]
        accuracy += [torch.sum(diff == 0)]

    return accuracy

def prediction_stat(outputs, labels, n_classes):
    lbl = labels.data
    valid = lbl < n_classes

    classwise_pixel_acc = []
    classwise_gtpixels = []
    classwise_predpixels = []
    for output in outputs:
        _, pred = output.data.max(dim=1)
        for m in range(n_classes):
            mask1 = lbl == m
            mask2 = pred[valid] == m
            diff = pred[mask1] - lbl[mask1]
            classwise_pixel_acc += [torch.sum(diff == 0)]
            classwise_gtpixels += [torch.sum(mask1)]
            classwise_predpixels += [torch.sum(mask2)]

    return classwise_pixel_acc, classwise_gtpixels, classwise_predpixels

def prediction_stat_confusion_matrix(logits, annotation, n_classes):
    labels = range(n_classes)

    # First we do argmax on gpu and then transfer it to cpu
    logits = logits.data
    annotation = annotation.data
    _, prediction = logits.max(1)
    prediction = prediction.squeeze(1)

    prediction_np = prediction.cpu().numpy().flatten()
    annotation_np = annotation.cpu().numpy().flatten()

    # Mask-out value is ignored by default in the sklearn
    # read sources to see how that was handled
    current_confusion_matrix = confusion_matrix(y_true=annotation_np,
                                                y_pred=prediction_np,
                                                labels=labels)

    return current_confusion_matrix