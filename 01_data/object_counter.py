import zarr
import glob
import sys
import os
import numpy as np

def count_pixels(arr):

    print("shape: ",arr.shape)
    uniques = np.unique(arr)
    print("number of objects ", len(uniques))

    non_zero_count = np.sum(arr > 0) #gets all pixels above 0
    all_count = np.prod(arr.shape) #multiplication product of all array dimensions
    #subtracting 1 because 0 value (i.e. not an object) is counted as a unique object
    avg_size_px = all_count/(len(uniques)-1)
    avg_vol = avg_size_px * (70*3*3)
    print("Avg vol of objects: ", avg_vol)
    print("Avg num voxels of objects: ", avg_size_px)
    print("non zero ", non_zero_count)
    print("all count", all_count)
    print("annotated ratio ", non_zero_count/all_count)

if __name__ == "__main__":

    input_zarr = sys.argv[1]

    f = zarr.open(input_zarr,"r")
    print(" ")
    print(input_zarr)
    for i in glob.glob(os.path.join(input_zarr,"*")):
        ds_name = i.split("/")[-1]
        print(ds_name)
        if "image" not in ds_name or ".z" not in ds_name:
            arr = f[ds_name][:] #changes to numpy array
            count_pixels(arr)
