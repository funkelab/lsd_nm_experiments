from __future__ import print_function
import sys
from gunpowder import *
from gunpowder.tensorflow import *
import malis
import glob
import os
import math
import json
import tensorflow as tf
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

data_dir = "../../../01_data/funke/zebrafinch/training"

samples = glob.glob(os.path.join(data_dir, "*.zarr"))

neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

phase_switch = 10000

with open("train_net.json", "r") as f:
    config = json.load(f)


def add_malis_loss(graph):
    affs = graph.get_tensor_by_name(config["affs"])
    gt_affs = graph.get_tensor_by_name(config["gt_affs"])
    gt_seg = tf.placeholder(tf.int64, shape=(48, 56, 56), name="gt_seg")
    gt_affs_mask = tf.placeholder(tf.int64, shape=(3, 48, 56, 56), name="gt_affs_mask")

    loss = malis.malis_loss_op(affs, gt_affs, gt_seg, neighborhood, gt_affs_mask)
    malis_summary = tf.summary.scalar("setup07_malis_loss", loss)
    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8,
        name="malis_optimizer",
    )

    optimizer = opt.minimize(loss)

    return (loss, optimizer)


def train_until(max_iteration):
    if tf.train.latest_checkpoint("."):
        trained_until = int(tf.train.latest_checkpoint(".").split("_")[-1])
    else:
        trained_until = 0
    if trained_until >= max_iteration:
        return

    if trained_until < phase_switch and max_iteration > phase_switch:
        train_until(phase_switch)

    phase = "euclid" if max_iteration <= phase_switch else "malis"
    print("Training in phase %s until %i" % (phase, max_iteration))

    raw = ArrayKey("RAW")
    labels = ArrayKey("GT_LABELS")
    labels_mask = ArrayKey("GT_LABELS_MASK")
    affs = ArrayKey("PREDICTED_AFFS")
    gt = ArrayKey("GT_AFFINITIES")
    gt_affs_mask = ArrayKey("GT_AFFINITIES_MASK")
    gt_affs_scale = ArrayKey("GT_AFFINITIES_SCALE")
    affs_gradient = ArrayKey("AFFS_GRADIENT")

    voxel_size = Coordinate((20, 9, 9))
    input_size = Coordinate(config["input_shape"]) * voxel_size
    output_size = Coordinate(config["output_shape"]) * voxel_size

    # max labels padding calculated
    labels_padding = Coordinate((500, 369, 369))

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(gt, output_size)
    request.add(gt_affs_mask, output_size)

    if phase is "euclid":
        request.add(gt_affs_scale, output_size)

    snapshot_request = BatchRequest({affs: request[gt], affs_gradient: request[gt]})

    data_sources = tuple(
        ZarrSource(
            sample,
            datasets={
                raw: "volumes/raw",
                labels: "volumes/labels/neuron_ids",
                labels_mask: "volumes/labels/labels_mask",
            },
            array_specs={
                raw: ArraySpec(interpolatable=True),
                labels: ArraySpec(interpolatable=False),
                labels_mask: ArraySpec(interpolatable=False),
            },
        )
        + Normalize(raw)
        + Pad(raw, None)
        + Pad(labels, labels_padding)
        + Pad(labels_mask, labels_padding)
        + RandomLocation(min_masked=0.5, mask=labels_mask)
        for sample in samples
    )

    train_pipeline = data_sources
    train_pipeline += RandomProvider()
    train_pipeline += ElasticAugment(
        control_point_spacing=[4, 4, 10],
        jitter_sigma=[0, 2, 2],
        rotation_interval=[0, math.pi / 2.0],
        prob_slip=0.05,
        prob_shift=0.05,
        max_misalign=10,
        subsample=8,
    )
    train_pipeline += SimpleAugment(transpose_only=[1, 2])
    train_pipeline += IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True)
    train_pipeline += GrowBoundary(labels, labels_mask, steps=1, only_xy=True)

    if phase == "malis":
        train_pipeline += RenumberConnectedComponents(labels=labels)

    train_pipeline += AddAffinities(
        neighborhood,
        labels=labels,
        labels_mask=labels_mask,
        affinities=gt,
        affinities_mask=gt_affs_mask,
    )

    if phase == "euclid":
        train_pipeline += BalanceLabels(gt, gt_affs_scale, gt_affs_mask)

    train_pipeline += IntensityScaleShift(raw, 2, -1)
    train_pipeline += PreCache(cache_size=40, num_workers=10)

    train_inputs = {
        config["raw"]: raw,
        config["gt_affs"]: gt,
    }
    if phase == "euclid":
        train_loss = config["loss"]
        train_optimizer = config["optimizer"]
        train_summary = config["summary"]
        train_inputs[config["affs_loss_weights"]] = gt_affs_scale
    else:
        train_loss = None
        train_optimizer = add_malis_loss
        train_inputs["gt_seg:0"] = labels
        train_inputs["gt_affs_mask:0"] = gt_affs_mask
        train_summary = "setup07_malis_loss:0"

    train_pipeline += Train(
        "train_net",
        optimizer=train_optimizer,
        loss=train_loss,
        inputs=train_inputs,
        outputs={config["affs"]: affs},
        gradients={config["affs"]: affs_gradient},
        summary=train_summary,
        log_dir="log",
        save_every=10000,
    )

    train_pipeline += IntensityScaleShift(raw, 0.5, 0.5)
    train_pipeline += Snapshot(
        {
            raw: "volumes/raw",
            labels: "volumes/labels/neuron_ids",
            gt: "volumes/gt_affinities",
            affs: "volumes/pred_affinities",
            gt_affs_mask: "volumes/labels/gt_affs_mask",
            labels_mask: "volumes/labels/mask",
            affs_gradient: "volumes/affs_gradient",
        },
        dataset_dtypes={labels: np.uint64},
        every=1000,
        output_filename="batch_{iteration}.hdf",
        additional_request=snapshot_request,
    )

    train_pipeline += PrintProfilingStats(every=10)

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration - trained_until):
            b.request_batch(request)
    print("Training finished")


if __name__ == "__main__":
    iteration = int(sys.argv[1])
    train_until(iteration)
