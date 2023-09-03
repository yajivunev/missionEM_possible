import zarr
import os
import sys

if __name__ == "__main__":

    input_zarr = sys.argv[1]
    split_z = int(sys.argv[2])

    f = zarr.open(input_zarr,"r")
    out1 = zarr.open("1"+input_zarr,"a")
    out2 = zarr.open("2"+input_zarr,"a")

    for ds_name in ["image", "endosomes_corrected", "lysosomes_corrected", "mask", "endosomes", "lysosomes"]:

        z1 = f[ds_name][:split_z]
        z2 = f[ds_name][split_z:]

        print(f"writing {ds_name}")
        out1[ds_name] = z1
        out1[ds_name].attrs["resolution"] = [70,3,3]
        out2[ds_name] = z2
        out2[ds_name].attrs["resolution"] = [70,3,3]
