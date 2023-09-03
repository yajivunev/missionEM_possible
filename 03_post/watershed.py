import zarr
import numpy as np
import sys
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt, maximum_filter
from skimage.measure import label


def watershed_from_boundary_distance(
        boundary_distances: np.ndarray,
        boundary_mask: np.ndarray,
        id_offset: float = 0,
        min_seed_distance: int = 10
        ):
    """Function to compute a watershed from boundary distances."""

    # get our seeds 
    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered==boundary_distances
    seeds = label(maxima)

    seeds[seeds!=0] += id_offset

    # calculate our segmentation
    segmentation = watershed(
        boundary_distances.max() - boundary_distances,
        seeds,
        mask=boundary_mask)
    
    return segmentation

if __name__ == "__main__":

    pred_zarr = sys.argv[1]
    pred_ds = sys.argv[2]
    try:
        thresh = float(sys.argv[3])
    except:
        thresh = 0.66

    pred = zarr.open(pred_zarr,"r")[pred_ds][:]

    labels = np.zeros(pred.shape[-3:], dtype=np.uint32)
    depth = labels.shape[0]
    
    mean_pred = 0.5 * (pred[1] + pred[2])

    for z in range(depth):
        
        boundary_mask = mean_pred[z] > 0.5 * 255
        boundary_distances = distance_transform_edt(boundary_mask)

        labels[z] = watershed_from_boundary_distance(
            boundary_distances,
            boundary_mask
        ).astype(np.uint32)

    f = zarr.open(pred_zarr,"a")
    f[f'{pred_ds}_watershed_labels'] = labels
    f[f'{pred_ds}_watershed_labels'].attrs["offset"] = [0,0,0]
    f[f'{pred_ds}_watershed_labels'].attrs["resolution"] = [70,24,24]
