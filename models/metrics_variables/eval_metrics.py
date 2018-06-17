import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import csv

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

individual = False

if (individual == True):
    # with open('metrics_variables_belt_test_exp4_thr025.pkl', 'rb') as f:
    with open(os.path.join('umbrales equiespaciados exp4', 'thr099.pkl'), 'rb') as f:
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

    f.close()

else:
    ap_per_thr = []
    precision_per_thr = []
    recall_per_thr = []
    tp_per_thr = []
    fp_per_thr = []
    fn_per_thr = []
    tpr_per_thr = []
    fnr_per_thr = []
    f1_per_thr = []

    eval_thr = ['0', '01', '02', '03', '04', '05', '06', '07', '08', '09', '099']

    for i, thr_file in enumerate(eval_thr):

        with open(os.path.join('umbrales equiespaciados exp4', 'thr'+thr_file+'.pkl'), 'rb') as f:
            scores_per_class, tp_fp_labels_per_class, num_gt_instances, num_class, ap_per_class, \
            mean_ap, precision_per_class, recalls_per_class, corloc_per_class, mean_corloc = pickle.load(f)

        tp_fp_labels = np.concatenate(tp_fp_labels_per_class[0])
        pr = precision_per_class[0][-1]
        rc = recalls_per_class[0][-1]
        precision_per_thr.append(pr)
        recall_per_thr.append(rc)

        ap_per_thr.append(ap_per_class[0])
        tp_per_thr.append(sum(tp_fp_labels.astype(int)))
        fp_per_thr.append(len(tp_fp_labels) - tp_per_thr[i])
        fn_per_thr.append(num_gt_instances[0] - tp_per_thr[i])

        tpr_per_thr.append(rc)
        fnr_per_thr.append(1 - tpr_per_thr[i])
        # TNR = TN / (TN + FP)
        # FPR = FP / (TN + FP)

        f1_per_thr.append(2 * (pr * rc) / (pr + rc))

        plt.plot(recalls_per_class[0], precision_per_class[0])
        f.close()

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(['0', '01', '02', '03', '04', '05', '06', '07', '08', '09', '099'])
    plt.show()

    with open('metrics_exp4.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # spamwriter.writerow(['AP'])
        spamwriter.writerow(ap_per_thr)
        # spamwriter.writerow(['Precision'])
        spamwriter.writerow(precision_per_thr)
        # spamwriter.writerow('Recall')
        spamwriter.writerow(recall_per_thr)
        # spamwriter.writerow('FNR')
        spamwriter.writerow(fnr_per_thr)
        # spamwriter.writerow('F1')
        spamwriter.writerow(f1_per_thr)
        # spamwriter.writerow('TP')
        spamwriter.writerow(tp_per_thr)
        # spamwriter.writerow('FP')
        spamwriter.writerow(fp_per_thr)
        # spamwriter.writerow('FN')
        spamwriter.writerow(fn_per_thr)
    csvfile.close()