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

    out_ds_endo = f'pred_endo_affs_{iteration}'
    out_ds_lyso = f'pred_lyso_affs_{iteration}'

    model = Model()
    model.eval()
    
    input_shape = gp.Coordinate((12, 260, 260))
    output_shape = gp.Coordinate((4, 220, 220))
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
    pred_endo = gp.ArrayKey("PRED_ENDO_AFFS")
    pred_lyso = gp.ArrayKey("PRED_LYSO_AFFS")

    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(pred_endo, output_size)
    scan_request.add(pred_lyso, output_size)

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
        total_output_roi = source.spec[raw].roi
        total_input_roi = source.spec[raw].roi.grow(context,context)

    prepare_ds(
            out_file,
            out_ds_endo,
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
            out_ds_lyso,
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

    unsqueeze = gp.Unsqueeze([raw], axis=0)
    unsqueeze += gp.Unsqueeze([raw], axis=0)

    predict = gp.torch.Predict(
            model,
            checkpoint=f'model_checkpoint_{iteration}',
            inputs = {
                'input': raw
            },
            outputs = {
                0: pred_endo,
                1: pred_lyso,
            })


    scan = gp.Scan(scan_request)

    write = gp.ZarrWrite(
            dataset_names={
                pred_endo: out_ds_endo,
                pred_lyso: out_ds_lyso,
            },
            store=out_file)

    pipeline = (
            source +
            gp.Normalize(raw) +
            gp.Pad(raw, None) +
            gp.Unsqueeze([raw]) +
            gp.Unsqueeze([raw]) +
            predict +
            gp.Squeeze([raw,pred_endo,pred_lyso]) +
            gp.Squeeze([raw]) +
            gp.IntensityScaleShift(pred_endo, 255, 0) +
            gp.IntensityScaleShift(pred_lyso, 255, 0) +
            write+
            scan)

    predict_request = gp.BatchRequest()
    predict_request[raw] = total_input_roi
    predict_request[pred_endo] = total_output_roi
    predict_request[pred_lyso] = total_output_roi

    with gp.build(pipeline):
        pipeline.request_batch(predict_request)

    return total_output_roi

if __name__ == "__main__":

    iteration = sys.argv[1]
    raw_file = sys.argv[2]
    raw_ds = sys.argv[3]
    out_file = sys.argv[4]

    predict(iteration,raw_file,raw_ds,out_file)
