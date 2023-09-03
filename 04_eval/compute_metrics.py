import os
import seaborn as sns
import numpy as np
import sys
import zarr
import matplotlib.pyplot as plt
from funlib.evaluate import rand_voi

def compute_iou(arr1, arr2):

    intersection = np.logical_and(arr1, arr2)

    union = np.logical_or(arr1, arr2)

    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score


def compute_3x3_confusion_matrix(gt_endo, pred_endo, gt_lyso, pred_lyso):

    TN = np.sum((gt_endo == 0) & (pred_endo == 0) & (gt_lyso == 0) & (pred_lyso == 0))
   
    FP_e = np.sum((gt_endo == 0) & (pred_endo == 1))
    FP_l = np.sum((gt_lyso == 0) & (pred_lyso == 1))
    
    FN_e = np.sum((gt_endo == 1) & (pred_endo == 0))
    FN_l = np.sum((gt_lyso == 1) & (pred_lyso == 0))

    TP_e = np.sum((gt_endo == 1) & (pred_endo == 1))
    TP_l = np.sum((gt_lyso == 1) & (pred_lyso == 1))

    Mis_e_l = np.sum((gt_endo == 1) & (pred_lyso == 1))
    Mis_l_e = np.sum((gt_lyso == 1) & (pred_endo == 1))
   
    return np.array([[TN, FP_e, FP_l], 
                     [FN_e, TP_e, Mis_e_l], 
                     [FN_l, Mis_l_e, TP_l]])

# Function to compute all metrics: IoU, Precision-Recall curve, and Average IoU
def compute_metrics(ground_truth, prediction):
    iou_list, precision_list, recall_list = [], [], []
    
    # Compute IoU and Average IoU
    for gt_label in np.unique(ground_truth):
        if gt_label == 0: continue
        gt_instance = (ground_truth == gt_label)
        best_iou_gt = 0
        for pred_label in np.unique(prediction):
            if pred_label == 0: continue
            pred_instance = (prediction == pred_label)
            iou_score = compute_iou(gt_instance, pred_instance)
            if iou_score > best_iou_gt:
                best_iou_gt = iou_score
        iou_list.append(best_iou_gt)

    for pred_label in np.unique(prediction):
        if pred_label == 0: continue
        pred_instance = (prediction == pred_label)
        best_iou_pred = 0
        for gt_label in np.unique(ground_truth):
            if gt_label == 0: continue
            gt_instance = (ground_truth == gt_label)
            iou_score = compute_iou(gt_instance, pred_instance)
            if iou_score > best_iou_pred:
                best_iou_pred = iou_score
        iou_list.append(best_iou_pred)

    average_iou = np.mean(iou_list)
    
    # Compute Precision-Recall curve
    for iou_threshold in np.arange(0.5, 1.0, 0.05):
        TP, FP, FN = 0, 0, 0
        for iou_score in iou_list:
            if iou_score > iou_threshold:
                TP += 1
            else:
                FP += 1
                FN += 1

        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
    
    return average_iou, precision_list, recall_list


if __name__ == "__main__":

    pred_zarr = sys.argv[1]
    pred_endo_labels_ds = sys.argv[2]
    pred_lyso_labels_ds = sys.argv[3]
   
    zarr_vol = pred_zarr.split('/')[-1].split('.')[0]
    setup = pred_zarr.split('/')[-2]
    gt_zarr = os.path.join('../01_data/',os.path.basename(pred_zarr))

    pred_endo_labels = zarr.open(pred_zarr,"r")[pred_endo_labels_ds][:]
    pred_lyso_labels = zarr.open(pred_zarr,"r")[pred_lyso_labels_ds][:]
    
    gt_endo = zarr.open(gt_zarr,"r")["endosomes_corrected/s2"][:]
    gt_lyso = zarr.open(gt_zarr,"r")["lysosomes_corrected/s2"][:]
    gt_mask = zarr.open(gt_zarr,"r")["mask/s2"][:]

    pred_endo_labels *= gt_mask
    pred_lyso_labels *= gt_mask

    #IOU + PR
    average_iou_endo, precision_list_endo, recall_list_endo = compute_metrics(gt_endo, pred_endo_labels)
    average_iou_lyso, precision_list_lyso, recall_list_lyso = compute_metrics(gt_lyso, pred_lyso_labels)

    #compute
    endo_voi = rand_voi(
        gt_endo.astype(np.uint64),
        pred_endo_labels.astype(np.uint64),
        return_cluster_scores=False)
    
    lyso_voi = rand_voi(
        gt_lyso.astype(np.uint64),
        pred_lyso_labels.astype(np.uint64),
        return_cluster_scores=False)

    endo_voi['voi_sum'] = endo_voi['voi_split']+endo_voi['voi_merge']
    endo_voi['nvi_sum'] = endo_voi['nvi_split']+endo_voi['nvi_merge']

    lyso_voi['voi_sum'] = lyso_voi['voi_split']+lyso_voi['voi_merge']
    lyso_voi['nvi_sum'] = lyso_voi['nvi_split']+lyso_voi['nvi_merge']

    for k in {'voi_split_i', 'voi_merge_j'}:
        del endo_voi[k]
        del lyso_voi[k]

    print(f"Average IoU endo = {average_iou_endo}; VoI sum endo = {endo_voi['voi_sum']}")
    print(f"Average IoU lyso = {average_iou_lyso}; Voi sum lyso = {lyso_voi['voi_sum']}")

    # save PR curves
    plt.figure()
    plt.plot(recall_list_endo, precision_list_endo, marker='.', label='Endo')
    plt.plot(recall_list_lyso, precision_list_lyso, marker='.', label='Lyso')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"results/PR_curve_{zarr_vol}_{setup}_{pred_endo_labels_ds}_{pred_lyso_labels_ds}.png")

    cm_3x3 = compute_3x3_confusion_matrix(
        (gt_endo > 0).astype(int), (pred_endo_labels > 0).astype(int), 
        (gt_lyso > 0).astype(int), (pred_lyso_labels > 0).astype(int))

    # Plot and Save 3x3 Confusion Matrix
    plt.figure()
    sns.heatmap(cm_3x3, annot=True, fmt='g', cmap='Blues', 
                xticklabels=['TN', 'FP_e', 'FP_l'], 
                yticklabels=['TN', 'FN_e', 'FN_l'])

    plt.title('Confusion Matrix for Endo and Lyso and background')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f"results/CM_{zarr_vol}_{setup}_{pred_endo_labels_ds}_{pred_lyso_labels_ds}.png")
