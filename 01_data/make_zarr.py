import glob
import sys
import os
import zarr
import numpy as np
import tifffile

def return_array_from_tif(tif_path):

    with tifffile.TiffFile(tif_path) as tif:
        input_array = tif.asarray()

    return input_array


if __name__ == "__main__":

    input_stacks = sys.argv[1:]

    for thing in input_stacks:

        out_container = thing.split('/')[1]+".zarr"
        f = zarr.open(out_container, "a")

        for raw_tiff in glob.glob(os.path.join(thing,"*corrected*.tif")) + glob.glob(os.path.join(thing,"*mask*.tif")):

            arr = return_array_from_tif(raw_tiff)

            if "some" in raw_tiff:
                arr = arr.astype(np.uint32)
            
            if 'mask' in raw_tiff:
                arr = (arr > 0).astype(np.uint8)

            ds_name = os.path.basename(raw_tiff).split('.')[0]

            print(f"writing {ds_name} into {out_container}!!!")
            f[ds_name] = arr
            f[ds_name].attrs["resolution"] = [70,3,3]
