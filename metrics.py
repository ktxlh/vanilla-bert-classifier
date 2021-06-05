import torch

REAL, FAKE = 1, 0
def evaluate(targets, preds, prints):
    targets = torch.tensor(targets, dtype=torch.long)
    preds = torch.tensor(preds, dtype=torch.long)
    
    acc = sum(targets == preds).item() / len(targets)
    if prints: print("Accuracy {:.4f}".format(acc))
    ret = [acc]
    for name, label in [('Fake', FAKE), ('Real', REAL)]:
        correct_true = sum((targets == label) * (preds == label)).item()
        target_true = sum(targets == label).item()
        predicted_true = sum(preds == label).item()
        if correct_true == 0:
            precision = 0.0
            recall = 0.0
            f1_score = 0.0
        else:
            precision = correct_true / predicted_true
            recall = correct_true / target_true
            f1_score = 2 * precision * recall / (precision + recall)
        if prints: print("{} precision {:.4f}".format(name, precision))
        if prints: print("{} recall    {:.4f}".format(name, recall))
        if prints: print("{} f1_score  {:.4f}".format(name, f1_score))
        ret.extend([precision, recall, f1_score])
    return ret