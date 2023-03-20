from __future__ import print_function
import sys
from gunpowder import *
from gunpowder.tensorflow import *
from mala.gunpowder import AddLocalShapeDescriptor
import os
import glob
import math
import json
import tensorflow as tf
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

data_dir = "../../../01_data/funke/zebrafinch/training"

samples = glob.glob(os.path.join(data_dir, "*.zarr"))


def train_until(max_iteration):
    if tf.train.latest_checkpoint("."):
        trained_until = int(tf.train.latest_checkpoint(".").split("_")[-1])
    else:
        trained_until = 0
    if trained_until >= max_iteration:
        return

    with open("train_net.json", "r") as f:
        config = json.load(f)

    raw = ArrayKey("RAW")
    labels = ArrayKey("GT_LABELS")
    labels_mask = ArrayKey("GT_LABELS_MASK")
    embedding = ArrayKey("PREDICTED_EMBEDDING")
    gt_embedding = ArrayKey("GT_EMBEDDING")
    embedding_gradient = ArrayKey("EMBEDDING_GRADIENT")

    voxel_size = Coordinate((20, 9, 9))
    input_size = Coordinate(config["input_shape"]) * voxel_size
    output_size = Coordinate(config["output_shape"]) * voxel_size

    # max labels padding calculated
    labels_padding = Coordinate((840, 720, 720))

    request = BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(gt_embedding, output_size)

    snapshot_request = BatchRequest(
        {embedding: request[gt_embedding], embedding_gradient: request[gt_embedding]}
    )

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

    train_pipeline = (
        data_sources
        + RandomProvider()
        + ElasticAugment(
            control_point_spacing=[4, 4, 10],
            jitter_sigma=[0, 2, 2],
            rotation_interval=[0, math.pi / 2.0],
            prob_slip=0.05,
            prob_shift=0.05,
            max_misalign=10,
            subsample=8,
        )
        + SimpleAugment(transpose_only=[1, 2])
        + IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1, z_section_wise=True)
        + AddLocalShapeDescriptor(labels, gt_embedding, sigma=120, downsample=2)
        + IntensityScaleShift(raw, 2, -1)
        + PreCache(cache_size=40, num_workers=10)
        + Train(
            "train_net",
            optimizer=config["optimizer"],
            loss=config["loss"],
            inputs={
                config["raw"]: raw,
                config["gt_embedding"]: gt_embedding,
                config["loss_weights_embedding"]: labels_mask,
            },
            outputs={config["embedding"]: embedding},
            gradients={config["embedding"]: embedding_gradient},
            summary=config["summary"],
            log_dir="log",
            save_every=10000,
        )
        + IntensityScaleShift(raw, 0.5, 0.5)
        + Snapshot(
            {
                raw: "volumes/raw",
                labels: "volumes/labels/neuron_ids",
                gt_embedding: "volumes/gt_embedding",
                embedding: "volumes/pred_embedding",
                labels_mask: "volumes/labels/mask",
                embedding_gradient: "volumes/embedding_gradient",
            },
            dataset_dtypes={labels: np.uint64},
            every=1000,
            output_filename="batch_{iteration}.hdf",
            additional_request=snapshot_request,
        )
        + PrintProfilingStats(every=10)
    )

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration - trained_until):
            b.request_batch(request)
    print("Training finished")


if __name__ == "__main__":
    iteration = int(sys.argv[1])
    train_until(iteration)
