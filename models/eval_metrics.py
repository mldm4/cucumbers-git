import pickle
from matplotlib import pyplot as plt
import numpy as np

#cálculo AP
# recall = np.concatenate([[0], recall, [1]])
#   precision = np.concatenate([[0], precision, [0]])
#
#   # Preprocess precision to be a non-decreasing array
#   for i in range(len(precision) - 2, -1, -1):
#     precision[i] = np.maximum(precision[i], precision[i + 1])
#
#   indices = np.where(recall[1:] != recall[:-1])[0] + 1
#   average_precision = np.sum(
#       (recall[indices] - recall[indices - 1]) * precision[indices])


with open('metrics_variables_test_new_exp1.pkl', 'rb') as f:
    scores_per_class, tp_fp_labels_per_class, num_gt_instances, num_class, ap_per_class, \
    mean_ap, precision_per_class, recalls_per_class, corloc_per_class, mean_corloc= pickle.load(f)

scores = scores_per_class[0]
tp_fp_labels = np.concatenate(tp_fp_labels_per_class[0])
precision = precision_per_class[0]
recall = recalls_per_class[0]
plt.plot(recall, precision)
plt.xlabel('recall')
plt.ylabel('precision')
plt.show()

labels = tp_fp_labels.astype(int)
tp_labels = sum(labels)
fp_labels = len(tp_fp_labels) - tp_labels
fn_labels = num_gt_instances - tp_labels

TPR = recall[-1]
#TNR = TN / (TN + FP)
#FPR = FP / (TN + FP)
FNR = 1-TPR

print("num_gt_instances", num_gt_instances)
print("num_class", num_class)
print("num_tp_labels", tp_labels)
print("num_fp_labels", fp_labels)
print("num_fn_labels", fn_labels)
print("ap_per_class", ap_per_class)
print("mean_ap", mean_ap)
print("precision", precision[-1])
print("recall", recall[-1])
print("FNR", FNR)
print("corloc_per_class", corloc_per_class)
print("mean_corloc", mean_corloc)
