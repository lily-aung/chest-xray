import torch

def confusion_matrix_torch(y_true, y_pred, num_classes):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[int(t), int(p)] += 1
    return cm

def per_class_precision_recall_f1(cm, eps=1e-12):
    cm = cm.to(torch.float32)
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    support = cm.sum(dim=1)
    return precision, recall, f1, support
def macro_f1_from_cm(cm):
    _, _, f1, _ = per_class_precision_recall_f1(cm)
    return float(f1.mean().item())
