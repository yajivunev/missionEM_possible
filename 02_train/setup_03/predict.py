import sys
import os
import math
import numpy as np
import torch
import gunpowder as gp
from funlib.persistence import prepare_ds
import logging

from model import Model

logging.basicConfig(level=logging.INFO)


def predict(iteration,raw_file,raw_ds,out_file):

    out_ds_endo_affs = f'pred_endo_affs_{iteration}'
    out_ds_lyso_affs = f'pred_lyso_affs_{iteration}'
    out_ds_endo_lsds = f'pred_endo_lsds_{iteration}'
    out_ds_lyso_lsds = f'pred_lyso_lsds_{iteration}'

    model = Model()
    model.eval()
    
    input_shape = gp.Coordinate((12, 260, 260))
    output_shape = gp.Coordinate((4, 168, 168))
#    increase = gp.Coordinate((20,200,200))
#    input_shape += increase
#    output_shape += increase

    print(output_shape)
    voxel_size = gp.Coordinate([70, 24, 24])
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size
    context = (input_size - output_size) // 2

    # pipeline
    raw = gp.ArrayKey("RAW")
    pred_endo_affs = gp.ArrayKey("PRED_ENDO_AFFS")
    pred_lyso_affs = gp.ArrayKey("PRED_LYSO_AFFS")
    pred_endo_lsds = gp.ArrayKey("PRED_ENDO_LSDS")
    pred_lyso_lsds = gp.ArrayKey("PRED_LYSO_LSDS")

    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(pred_endo_affs, output_size)
    scan_request.add(pred_lyso_affs, output_size)
    scan_request.add(pred_endo_lsds, output_size)
    scan_request.add(pred_lyso_lsds, output_size)

    source = gp.ZarrSource(
            raw_file,  # the zarr container
            {
                raw: raw_ds, 
            },
            {
                raw: gp.ArraySpec(interpolatable=True),
            }
        )

    with gp.build(source):
        total_input_roi = source.spec[raw].roi
        total_output_roi = source.spec[raw].roi.grow(-context,-context)

    prepare_ds(
            out_file,
            out_ds_endo_affs,
            gp.Roi(
                total_output_roi.get_offset(),
                total_output_roi.get_shape()
            ),
            voxel_size,
            np.uint8,
            write_size=output_size,
            compressor={'id': 'blosc', 'clevel': 3},
            delete=True,
            num_channels=3)

    prepare_ds(
            out_file,
            out_ds_lyso_affs,
            gp.Roi(
                total_output_roi.get_offset(),
                total_output_roi.get_shape()
            ),
            voxel_size,
            np.uint8,
            write_size=output_size,
            compressor={'id': 'blosc', 'clevel': 3},
            delete=True,
            num_channels=3)

    prepare_ds(
            out_file,
            out_ds_endo_lsds,
            gp.Roi(
                total_output_roi.get_offset(),
                total_output_roi.get_shape()
            ),
            voxel_size,
            np.uint8,
            write_size=output_size,
            compressor={'id': 'blosc', 'clevel': 3},
            delete=True,
            num_channels=10)

    prepare_ds(
            out_file,
            out_ds_lyso_lsds,
            gp.Roi(
                total_output_roi.get_offset(),
                total_output_roi.get_shape()
            ),
            voxel_size,
            np.uint8,
            write_size=output_size,
            compressor={'id': 'blosc', 'clevel': 3},
            delete=True,
            num_channels=10)

    unsqueeze = gp.Unsqueeze([raw], axis=0)
    unsqueeze += gp.Unsqueeze([raw], axis=0)

    predict = gp.torch.Predict(
            model,
            checkpoint=f'model_checkpoint_{iteration}',
            inputs = {
                'input': raw
            },
            outputs = {
                0: pred_endo_affs,
                1: pred_lyso_affs,
                2: pred_endo_lsds,
                3: pred_lyso_lsds,
            })


    scan = gp.Scan(scan_request)

    write = gp.ZarrWrite(
            dataset_names={
                pred_endo_affs: out_ds_endo_affs,
                pred_lyso_affs: out_ds_lyso_affs,
                pred_endo_lsds: out_ds_endo_lsds,
                pred_lyso_lsds: out_ds_lyso_lsds,
            },
            store=out_file)

    pipeline = (
            source +
            gp.Normalize(raw) +
            gp.Pad(raw, None) +
            gp.Unsqueeze([raw]) +
            gp.Unsqueeze([raw]) +
            predict +
            gp.Squeeze([raw,pred_endo_affs,pred_lyso_affs]) +
            gp.Squeeze([pred_endo_lsds,pred_lyso_lsds]) +
            gp.Squeeze([raw]) +
            gp.IntensityScaleShift(pred_endo_affs, 255, 0) +
            gp.IntensityScaleShift(pred_lyso_affs, 255, 0) +
            gp.IntensityScaleShift(pred_endo_lsds, 255, 0) +
            gp.IntensityScaleShift(pred_lyso_lsds, 255, 0) +
            write+
            scan)

    predict_request = gp.BatchRequest()
    predict_request[raw] = total_input_roi
    predict_request[pred_endo_affs] = total_output_roi
    predict_request[pred_lyso_affs] = total_output_roi
    predict_request[pred_endo_lsds] = total_output_roi
    predict_request[pred_lyso_lsds] = total_output_roi

    with gp.build(pipeline):
        pipeline.request_batch(predict_request)

    return total_output_roi

if __name__ == "__main__":

    iteration = sys.argv[1]
    raw_file = sys.argv[2]
    raw_ds = sys.argv[3]
    out_file = sys.argv[4]

    predict(iteration,raw_file,raw_ds,out_file)
