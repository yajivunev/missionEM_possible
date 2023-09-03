import numpy as np
import zarr
import sys

if __name__ == "__main__":

    input_zarr = sys.argv[1]

    f = zarr.open(input_zarr,"a")

    endo = f["endosomes_corrected"][:]
    lyso = f["lysosomes_corrected"][:]

    labels = np.stack([endo,lyso], axis=0)

    print(f"writing labels to {input_zarr}")

    f["labels"] = labels
    f["labels"].attrs["resolution"] = [70, 3, 3]
