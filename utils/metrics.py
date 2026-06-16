from torchmetrics.classification import MultilabelAUROC
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np

def measurement(outputs, labels, smooth=1e-10):
    tp, tn, fp, fn = 0, 0, 0, 0
    single_tp, single_tn, single_fp, single_fn = [0]*labels.shape[1], [0]*labels.shape[1], [0]*labels.shape[1], [0]*labels.shape[1]
    labels = labels.cpu().numpy()
    outputs = outputs.detach().cpu().clone().numpy()
    for j in range(labels.shape[0]):
        for i in range(labels.shape[1]):
            if (int(outputs[j][i]) == 1 and int(labels[j][i]) == 1):
                tp += 1
                single_tp[i] += 1
            if (int(outputs[j][i]) == 0 and int(labels[j][i]) == 0):
                tn += 1
                single_tn[i] += 1
            if (int(outputs[j][i]) == 1 and int(labels[j][i]) == 0):
                fp += 1
                single_fp[i] += 1
            if (int(outputs[j][i]) == 0 and int(labels[j][i]) == 1):
                fn += 1
                single_fn[i] += 1
    
    # if element is 0 in list, then repalce it with 1e-10
    # tp = tp if tp != 0 else smooth
    # tn = tn if tn != 0 else smooth
    # fp = fp if fp != 0 else smooth
    # fn = fn if fn != 0 else smooth
    # single_tp = [i if i != 0 else smooth for i in single_tp]
    # single_tn = [i if i != 0 else smooth for i in single_tn]
    # single_fp = [i if i != 0 else smooth for i in single_fp]
    # single_fn = [i if i != 0 else smooth for i in single_fn] 

    return tp, tn, fp, fn, single_tp, single_tn, single_fp, single_fn

def sub_measurement(single_tp, single_tn, single_fp, single_fn):
    acc, recall, precision, f1_score = [0]*len(single_tp), [0]*len(single_tp), [0]*len(single_tp), [0]*len(single_tp)

    for i in range(len(single_tp)):
        acc[i] = 0 if single_tp[i] == 1e-10 and single_tn[i] == 1e-10 and single_fp[i] == 1e-10 and single_fn[i] == 1e-10 else (single_tp[i]+single_tn[i]) / (single_tp[i]+single_tn[i]+single_fp[i]+single_fn[i]) * 100
        recall[i] = 0 if single_tp[i] == 1e-10 and single_fn[i] == 1e-10 else single_tp[i] / (single_tp[i]+single_fn[i])
        precision[i] = 0 if single_tp[i] == 1e-10 and single_fp[i] == 1e-10 else single_tp[i] / (single_tp[i]+single_fp[i])
        f1_score[i] = 0 if single_tp[i] == 1e-10 and single_fp[i] == 1e-10 and single_fn[i] == 1e-10 else (2*single_tp[i]) / (2*single_tp[i]+single_fp[i]+single_fn[i])

    return acc, recall, precision, f1_score

def auc_roc_curve(preds, targets, num_classes):
    AUC = MultilabelAUROC(num_labels=num_classes, average=None, thresholds=None)
    mAUC = MultilabelAUROC(num_labels=num_classes, average='macro', thresholds=None)
    return AUC(preds, targets), mAUC(preds, targets)

def compute_class_freqs(labels):
    N = len(labels)
    positive_frequencies = (np.sum(labels, 0)) / N
    negative_frequencies = (1- positive_frequencies)
    return positive_frequencies, negative_frequencies