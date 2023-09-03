import os
import math
import numpy as np
import torch
import gunpowder as gp
import logging
from lsd.train.gp import AddLocalShapeDescriptor

from model import Model

logging.basicConfig(level=logging.INFO)

input_shape = (12, 260, 260)
voxel_size = gp.Coordinate([70,24,24])

sigma = 240

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


def calc_max_padding(output_size, voxel_size, sigma, mode="shrink"):

    method_padding = gp.Coordinate((sigma * 3,) * 3)

    diag = np.sqrt(output_size[1] ** 2 + output_size[2] ** 2)

    max_padding = gp.Roi(
        (
            gp.Coordinate([i / 2 for i in [output_size[0], diag, diag]])
            + method_padding
        ),
        (0,) * 3,
    ).snap_to_grid(voxel_size, mode=mode)

    return max_padding.get_begin()


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
            pred_endo_affs,
            gt_endo_affs,
            weights_endo_affs,
            pred_lyso_affs,
            gt_lyso_affs,
            weights_lyso_affs,
            pred_endo_lsds,
            gt_endo_lsds,
            weights_endo_lsds,
            pred_lyso_lsds,
            gt_lyso_lsds,
            weights_lyso_lsds,
        ):

        loss_endo_affs = self._calc_loss(pred_endo_affs, gt_endo_affs, weights_endo_affs)
        loss_lyso_affs = self._calc_loss(pred_lyso_affs, gt_lyso_affs, weights_lyso_affs)
        loss_endo_lsds = self._calc_loss(pred_endo_lsds, gt_endo_lsds, weights_endo_lsds)
        loss_lyso_lsds = self._calc_loss(pred_lyso_lsds, gt_lyso_lsds, weights_lyso_lsds)

        return loss_endo_affs + loss_lyso_affs + loss_endo_lsds + loss_lyso_lsds


def train_until(iterations):

    model = Model()
    
    output_shape = tuple(model(torch.zeros((1, 1) + input_shape))[0].shape[2:])
    print(output_shape)
    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size
    context = calc_max_padding(output_size, voxel_size, sigma)


    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = WeightedMSELoss()

    # pipeline
    raw = gp.ArrayKey("RAW")
    mask = gp.ArrayKey("MASK")
    labels_endo = gp.ArrayKey("LABELS_ENDO")
    labels_lyso = gp.ArrayKey("LABELS_LYSO")
    gt_endo_affs = gp.ArrayKey("GT_ENDO_AFFS")
    gt_lyso_affs = gp.ArrayKey("GT_LYSO_AFFS")
    gt_endo_affs_mask = gp.ArrayKey("GT_ENDO_AFFS_MASK")
    gt_lyso_affs_mask = gp.ArrayKey("GT_LYSO_AFFS_MASK")
    pred_endo_affs = gp.ArrayKey("PRED_ENDO_AFFS")
    pred_lyso_affs = gp.ArrayKey("PRED_LYSO_AFFS")
    weights_endo_affs = gp.ArrayKey("WEIGHTS_ENDO")
    weights_lyso_affs = gp.ArrayKey("WEIGHTS_LYSO")

    gt_endo_lsds = gp.ArrayKey("GT_ENDO_LSDS")
    gt_lyso_lsds = gp.ArrayKey("GT_LYSO_LSDS")
    gt_endo_lsds_mask = gp.ArrayKey("GT_ENDO_LSDS_MASK")
    gt_lyso_lsds_mask = gp.ArrayKey("GT_LYSO_LSDS_MASK")
    pred_endo_lsds = gp.ArrayKey("PRED_ENDO_LSDS")
    pred_lyso_lsds = gp.ArrayKey("PRED_LYSO_LSDS")

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
                raw: gp.ArraySpec(interpolatable=True),#, voxel_size=voxel_size),
                labels_endo: gp.ArraySpec(interpolatable=False),#, voxel_size=voxel_size),
                labels_lyso: gp.ArraySpec(interpolatable=False),#, voxel_size=voxel_size),
                mask: gp.ArraySpec(interpolatable=False),#, voxel_size=voxel_size),
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

    simple_augment = gp.SimpleAugment(transpose_only=[1,2])
    deform_augment = gp.ElasticAugment(
        control_point_spacing=(1, 24, 24),
        jitter_sigma=(0, 1, 1),
        rotation_interval=(0,math.pi/2),
        scale_interval=(0.9, 1.1),
        subsample=2)
    intensity_augment = gp.IntensityAugment(
        raw,
        scale_min=0.9,
        scale_max=1.1,
        shift_min=-0.01,
        shift_max=0.01)

    add_lsds = AddLocalShapeDescriptor(
            labels_endo,
            gt_endo_lsds,
            sigma=sigma,
            lsds_mask=gt_endo_lsds_mask,
            labels_mask=mask,
            downsample=2)
    add_lsds += AddLocalShapeDescriptor(
            labels_lyso,
            gt_lyso_lsds,
            sigma=sigma,
            lsds_mask=gt_lyso_lsds_mask,
            labels_mask=mask,
            downsample=2)

    erode_labels = gp.GrowBoundary(labels_endo, steps=1, only_xy=True)
    erode_labels += gp.GrowBoundary(labels_lyso, steps=1, only_xy=True)
    
    add_affs = gp.AddAffinities(
            affinity_neighborhood=[[-1,0,0],[0,-1,0],[0,0,-1]],
            labels=labels_endo,
            affinities=gt_endo_affs,
            labels_mask=mask,
            affinities_mask=gt_endo_affs_mask)
    add_affs += gp.AddAffinities(
            affinity_neighborhood=[[-1,0,0],[0,-1,0],[0,0,-1]],
            labels=labels_lyso,
            affinities=gt_lyso_affs,
            labels_mask=mask,
            affinities_mask=gt_lyso_affs_mask)

    balance_labels = gp.BalanceLabels(gt_endo_affs, weights_endo_affs, mask=gt_endo_affs_mask)
    balance_labels += gp.BalanceLabels(gt_lyso_affs, weights_lyso_affs, mask=gt_lyso_affs_mask)

    unsqueeze = gp.Unsqueeze([raw, gt_endo_affs, gt_lyso_affs, weights_endo_affs, weights_lyso_affs, gt_endo_lsds, gt_lyso_lsds, gt_endo_lsds_mask, gt_lyso_lsds_mask], axis=0)
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
          0: pred_endo_affs,
          1: pred_lyso_affs,
          2: pred_endo_lsds,
          3: pred_lyso_lsds,
        },
        loss_inputs = {
          0: pred_endo_affs,
          1: gt_endo_affs,
          2: weights_endo_affs,
          3: pred_lyso_affs,
          4: gt_lyso_affs,
          5: weights_lyso_affs,
          6: pred_endo_lsds,
          7: gt_endo_lsds,
          8: gt_endo_lsds_mask,
          9: pred_lyso_lsds,
          10: gt_lyso_lsds,
          11: gt_lyso_lsds_mask,
        },
        log_dir="log",
        save_every=1000)

    #squeeze = gp.Squeeze([raw, gt_endo_affs, gt_lyso_affs, weights_endo_affs, weights_lyso_affs, gt_endo_lsds, gt_lyso_lsds, gt_endo_lsds_mask, gt_lyso_lsds_mask], axis=0)
    squeeze = gp.Squeeze([raw, gt_endo_affs, gt_lyso_affs, gt_endo_lsds, gt_lyso_lsds], axis=0)
    squeeze = gp.Squeeze([raw, pred_endo_affs, pred_lyso_affs, pred_endo_lsds, pred_lyso_lsds], axis=0)
    #squeeze += gp.Squeeze([raw], axis=0)

    snapshot = gp.Snapshot(
        {
            raw: 'raw',
#            labels_endo: 'labels_endo',
#            labels_lyso: 'labels_lyso',
#            mask: 'mask',
#            weights_endo_affs: 'weights_endo_affs',
#            weights_lyso_affs: 'weights_lyso_affs',
            gt_endo_affs: 'gt_endo_affs',
            gt_lyso_affs: 'gt_lyso_affs',
            pred_endo_affs: 'pred_endo_affs',
            pred_lyso_affs: 'pred_lyso_affs',
            gt_endo_lsds: 'gt_endo_lsds',
            gt_lyso_lsds: 'gt_lyso_lsds',
            pred_endo_lsds: 'pred_endo_lsds',
            pred_lyso_lsds: 'pred_lyso_lsds',
#            gt_endo_lsds_mask: 'gt_endo_lsds_mask',
#            gt_lyso_lsds_mask: 'gt_lyso_lsds_mask',
        },
        output_dir='snapshots',
        output_filename='batch_{iteration}.zarr',
        every=1000)

    pipeline = (
        sources +
        simple_augment +
        deform_augment +
        intensity_augment +
        add_lsds +
        erode_labels +
        add_affs +
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
    request.add(gt_endo_affs, output_size)
    request.add(pred_endo_affs, output_size)
    request.add(weights_endo_affs, output_size)
    request.add(gt_lyso_affs, output_size)
    request.add(pred_lyso_affs, output_size)
    request.add(weights_lyso_affs, output_size)
    request.add(gt_endo_lsds, output_size)
    request.add(pred_endo_lsds, output_size)
    request.add(gt_endo_lsds_mask, output_size)
    request.add(gt_lyso_lsds, output_size)
    request.add(pred_lyso_lsds, output_size)
    request.add(gt_lyso_lsds_mask, output_size)

    with gp.build(pipeline):
        for i in range(iterations):
            batch = pipeline.request_batch(request)


if __name__ == "__main__":

    train_until(100000)
