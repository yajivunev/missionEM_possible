import numpy as np
# import zarr
# import os
# import blob
# import sys

def IoU(gt, pred):
    # Ensure the masks are binary (0 or 1)
    pred = np.where(pred > 0, 1, 0)
    gt = np.where(gt > 0, 1, 0)
    

    # Intersection and union
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()

    # Avoid division by zero
    if union == 0:
        return 0

    return intersection / union


if __name__ == "__main__":
    
    zarr_prediction = sys.argv[1]
    zarr_gt = sys.argv[2]

    zarr_prediction = zarr.open(zarr_prediction,"r")
    zarr_gt = zarr.open(zarr_gt,"r")

    lyso_pred = zarr_prediction["lysosomes"][:]
    endo_pred = zarr_prediction["endosomes"][:]
    lyso_gt = zarr_gt["lysosomes"][:]
    endo_gt = zarr_gt["endosomes"][:]

    endo_IoU = IoU(endo_gt, endo_pred)
    lyso_IoU = IoU(lyso_gt, lyso_pred)

    print("Endo IoU: ", endo_IoU)
    print("Lyso IoU: ", lyso_IoU)
