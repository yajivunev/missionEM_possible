import zarr
import numpy as np
import sys
from skimage.measure import label
from skimage.filters import threshold_otsu


def thresh_relabel(pred, thresh="otsu"):

    if thresh == "otsu":
        threshold_value = threshold_otsu(pred)
    else:
        threshold_value = thresh*255

    binarized_array = (pred > threshold_value)#.astype(np.uint32)

    labeled_array = label(binarized_array)

    return labeled_array


if __name__ == "__main__":

    pred_zarr = sys.argv[1]
    pred_ds = sys.argv[2]
    try:
        thresh = float(sys.argv[3])
    except:
        thresh = "otsu"

    pred = zarr.open(pred_zarr,"r")[pred_ds][:]

    labels = thresh_relabel(pred, thresh).astype(np.uint64)

    f = zarr.open(pred_zarr,"a")
    f[f'{pred_ds}_thresh_{thresh}_labels'] = labels[0]
    f[f'{pred_ds}_thresh_{thresh}_labels'].attrs["offset"] = [0,0,0]
    f[f'{pred_ds}_thresh_{thresh}_labels'].attrs["resolution"] = [70,24,24]
