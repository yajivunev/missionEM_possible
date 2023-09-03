import os
import math
import numpy as np
import torch
import gunpowder as gp
import logging

from model import Model

logging.basicConfig(level=logging.INFO)

input_shape = (12, 260, 260)
voxel_size = gp.Coordinate([70, 24, 24])

lr = 1e-4
batch_size = 1

data_dir = "../../01_data/"
training_zarrs = [
        "TRAIN_1.zarr",
        "TRAIN_2.zarr",
        "TRAIN_3.zarr",
        "TRAIN_4.zarr",
        "TRAIN_5.zarr",
]


class BinarizeLabels(gp.BatchFilter):
    def __init__(self, input_key: gp.ArrayKey, output_key: gp.ArrayKey):
        self.input_key = input_key
        self.output_key = output_key

    def setup(self):
        self.provides(self.output_key, self.spec[self.input_key].copy())

    def prepare(self, request: gp.BatchRequest) -> gp.BatchRequest:
        deps = gp.BatchRequest()
        deps[self.input_key] = request[self.output_key].copy()
        return deps

    def process(self, batch: gp.Batch, request: gp.BatchRequest) -> gp.Batch:
        outputs = gp.Batch()
        output_spec = batch[self.input_key].spec.copy()
        output_spec.dtype = np.uint8
        outputs[self.output_key] = gp.Array(
            (batch[self.input_key].data > 0).astype(np.uint8), output_spec
        )
        return outputs


class WeightedMSELoss(torch.nn.Module):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def _calc_loss(self, pred, target, weights):

        scale = (weights * (pred - target) ** 2)

        if len(torch.nonzero(scale)) != 0:

            mask = torch.masked_select(scale, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:

            loss = torch.mean(scale)

        return loss

    def forward(
            self,
            pred_endo,
            gt_endo,
            weights_endo,
            pred_lyso,
            gt_lyso,
            weights_lyso):

        loss_endo = self._calc_loss(pred_endo, gt_endo, weights_endo)
        loss_lyso = self._calc_loss(pred_lyso, gt_lyso, weights_lyso)

        return loss_endo + loss_lyso


def train_until(iterations):

    model = Model()
    
    output_shape = tuple(model(torch.zeros((1, 1) + input_shape))[0].shape[2:])
    print(output_shape)
    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size
    context = (input_size - output_size) * 2
    #context = gp.Coordinate((2000,2000,2000))


    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = WeightedMSELoss()

    # pipeline
    raw = gp.ArrayKey("RAW")
    mask = gp.ArrayKey("MASK")
    labels_endo = gp.ArrayKey("LABELS_ENDO")
    labels_lyso = gp.ArrayKey("LABELS_LYSO")
    gt_endo = gp.ArrayKey("GT_ENDO")
    gt_lyso = gp.ArrayKey("GT_LYSO")
    pred_endo = gp.ArrayKey("PRED_ENDO")
    pred_lyso = gp.ArrayKey("PRED_LYSO")
    weights_endo = gp.ArrayKey("WEIGHTS_ENDO")
    weights_lyso = gp.ArrayKey("WEIGHTS_LYSO")

    sources = tuple(
        gp.ZarrSource(
            os.path.join(data_dir, zarr_container),  # the zarr container
            {
                raw: 'image/s2', 
                labels_endo: 'endosomes_corrected/s2', 
                labels_lyso: 'lysosomes_corrected/s2', 
                mask: 'mask/s2'
            },
            {
                raw: gp.ArraySpec(interpolatable=True),
                labels_endo: gp.ArraySpec(interpolatable=False),
                labels_lyso: gp.ArraySpec(interpolatable=False),
                mask: gp.ArraySpec(interpolatable=False),
            }
        ) + 
        gp.Normalize(raw) +
        gp.Pad(raw, context) + 
        gp.Pad(labels_endo, context) +
        gp.Pad(labels_lyso, context) +
        gp.Pad(mask, context) +
        gp.RandomLocation(mask=mask, min_masked=0.5)
        for zarr_container in training_zarrs)

    sources += gp.RandomProvider()

    erode_labels = gp.GrowBoundary(labels_endo, steps=1, only_xy=True)
    erode_labels += gp.GrowBoundary(labels_lyso, steps=1, only_xy=True)
    binarize_labels = BinarizeLabels(labels_endo, gt_endo) 
    binarize_labels += BinarizeLabels(labels_lyso, gt_lyso) 

    simple_augment = gp.SimpleAugment(transpose_only=[1,2])
    deform_augment = gp.ElasticAugment(
        control_point_spacing=(1, 24, 24),
        jitter_sigma=(0, 1.0, 1.0),
        rotation_interval=(0,math.pi/2),
        scale_interval=(0.9, 1.1),
        subsample=4)
    intensity_augment = gp.IntensityAugment(
        raw,
        scale_min=0.9,
        scale_max=1.1,
        shift_min=-0.01,
        shift_max=0.01)

    balance_labels = gp.BalanceLabels(gt_endo, weights_endo, mask=mask)
    balance_labels += gp.BalanceLabels(gt_lyso, weights_lyso, mask=mask)

    unsqueeze = gp.Unsqueeze([raw, mask, gt_endo, gt_lyso, weights_endo, weights_lyso], axis=0)
    stack = gp.Stack(batch_size)
    precache = gp.PreCache(num_workers=4)

    train = gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs = {
          'input': raw
        },
        outputs = {
          0: pred_endo,
          1: pred_lyso
        },
        loss_inputs = {
          0: pred_endo,
          1: gt_endo,
          2: weights_endo,
          3: pred_lyso,
          4: gt_lyso,
          5: weights_lyso
        },
        log_dir="log",
        save_every=1000)

    squeeze = gp.Squeeze([raw, mask, gt_endo, gt_lyso, pred_endo, pred_lyso, weights_endo, weights_lyso], axis=0)
    squeeze += gp.Squeeze([raw, mask, gt_endo, gt_lyso, pred_endo, pred_lyso, weights_endo, weights_lyso], axis=0)

    snapshot = gp.Snapshot(
        {
            raw: 'raw',
            labels_endo: 'labels_endo',
            labels_lyso: 'labels_lyso',
            mask: 'mask',
            gt_endo: 'gt_endo',
            gt_lyso: 'gt_lyso',
            weights_endo: 'weights_endo',
            weights_lyso: 'weights_lyso',
            pred_endo: 'pred_endo',
            pred_lyso: 'pred_lyso'
        },
        output_dir='snapshots',
        output_filename='batch_{iteration}.zarr',
        every=1000)

    pipeline = (
        sources +
        erode_labels +
        binarize_labels +
        simple_augment +
        deform_augment +
        intensity_augment +
        balance_labels +
        unsqueeze +
        stack +
        precache +
        train +
        squeeze +
        snapshot)

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels_endo, output_size)
    request.add(labels_lyso, output_size)
    request.add(mask, output_size)
    request.add(gt_endo, output_size)
    request.add(pred_endo, output_size)
    request.add(weights_endo, output_size)
    request.add(gt_lyso, output_size)
    request.add(pred_lyso, output_size)
    request.add(weights_lyso, output_size)

    with gp.build(pipeline):
        for i in range(iterations):
            batch = pipeline.request_batch(request)


if __name__ == "__main__":

    train_until(10000)
