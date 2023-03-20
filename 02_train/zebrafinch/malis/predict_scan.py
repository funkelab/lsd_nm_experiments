from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
import json
import logging
import numpy as np
import os
import sys
import zarr

setup_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(setup_dir, "config.json"), "r") as f:
    net_config = json.load(f)

# voxels
input_shape = Coordinate(net_config["input_shape"])
output_shape = Coordinate(net_config["output_shape"])

# nm
voxel_size = Coordinate((20, 9, 9))
input_size = input_shape * voxel_size
output_size = output_shape * voxel_size


def predict(iteration, raw_file, raw_dataset, out_file, out_dataset):
    raw = ArrayKey("RAW")
    affs = ArrayKey("AFFS")

    scan_request = BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(affs, output_size)

    context = (input_size - output_size) / 2

    source = ZarrSource(
        raw_file,
        datasets={raw: raw_dataset},
        array_specs={
            raw: ArraySpec(interpolatable=True),
        },
    )

    with build(source):
        total_input_roi = source.spec[raw].roi
        total_output_roi = total_input_roi.grow(-context, -context)

    f = zarr.open(out_file, "w")

    ds = f.create_dataset(
        out_dataset,
        shape=[3] + list(total_output_roi.get_shape() / voxel_size),
        dtype=np.uint8,
    )

    ds.attrs["resolution"] = voxel_size
    ds.attrs["offset"] = total_output_roi.get_offset()

    pipeline = source

    pipeline += Pad(raw, size=None)

    pipeline += Normalize(raw)

    pipeline += IntensityScaleShift(raw, 2, -1)

    pipeline += Predict(
        os.path.join(setup_dir, "train_net_checkpoint_%d" % iteration),
        graph=os.path.join(setup_dir, "config.meta"),
        # max_shared_memory=(2*1024*1024*1024),
        inputs={net_config["raw"]: raw},
        outputs={net_config["affs"]: affs},
    )

    pipeline += IntensityScaleShift(affs, 255, 0)

    pipeline += ZarrWrite(
        dataset_names={
            affs: out_dataset,
        },
        output_filename=out_file,
    )

    pipeline += Scan(scan_request)

    predict_request = BatchRequest()

    predict_request.add(raw, total_input_roi.get_shape())
    predict_request.add(affs, total_output_roi.get_shape())

    print("Starting prediction...")
    with build(pipeline):
        pipeline.request_batch(predict_request)
    print("Prediction finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("gunpowder.nodes.hdf5like_write_base").setLevel(logging.DEBUG)

    iteration = 400000
    raw_file = "../../../01_data/funke/zebrafinch/training/gt_z255-383_y1407-1663_x1535-1791.zarr"
    raw_dataset = "volumes/raw"
    out_file = "test_prediction.zarr"
    out_dataset = "pred_affs"

    predict(iteration, raw_file, raw_dataset, out_file, out_dataset)
