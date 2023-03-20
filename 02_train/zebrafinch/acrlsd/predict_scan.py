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

with open(os.path.join(setup_dir, "test_net.json"), "r") as f:
    net_config = json.load(f)

# voxels
input_shape = Coordinate(net_config["input_shape"])
output_shape = Coordinate(net_config["output_shape"])

# nm
voxel_size = Coordinate((20, 9, 9))
input_size = input_shape * voxel_size
output_size = output_shape * voxel_size


def predict(
    iteration, raw_file, raw_dataset, auto_file, auto_dataset, out_file, out_dataset
):
    raw = ArrayKey("RAW")
    lsds = ArrayKey("PRETRAINED_LSDS")
    affs = ArrayKey("AFFS")

    print("Auto file is: %s, Auto dataset is: %s" % (auto_file, auto_dataset))

    scan_request = BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(lsds, input_size)
    scan_request.add(affs, output_size)

    context = (input_size - output_size) / 2

    raw_source = ZarrSource(
        raw_file,
        datasets={raw: raw_dataset},
        array_specs={raw: ArraySpec(interpolatable=True)},
    )

    lsd_source = ZarrSource(
        auto_file,
        datasets={lsds: auto_dataset},
        array_specs={lsds: ArraySpec(interpolatable=True)},
    )

    with build(lsd_source):
        total_input_roi = lsd_source.spec[lsds].roi.grow(context, context)
        total_output_roi = lsd_source.spec[lsds].roi

    f = zarr.open(out_file, "w")

    ds = f.create_dataset(
        out_dataset,
        shape=[3] + list(total_output_roi.get_shape() / voxel_size),
        dtype=np.uint8,
    )

    ds.attrs["resolution"] = voxel_size
    ds.attrs["offset"] = total_output_roi.get_offset()

    pipeline = (raw_source, lsd_source) + MergeProvider()

    pipeline += Pad(raw, None)
    pipeline += Normalize(raw)
    pipeline += IntensityScaleShift(raw, 2, -1)

    pipeline += Pad(lsds, context)
    pipeline += Normalize(lsds)

    pipeline += Predict(
        checkpoint=os.path.join(setup_dir, "train_net_checkpoint_%d" % iteration),
        graph=os.path.join(setup_dir, "test_net.meta"),
        # max_shared_memory=(2*1024*1024*1024),
        inputs={net_config["pretrained_lsd"]: lsds, net_config["raw"]: raw},
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
    predict_request.add(lsds, total_input_roi.get_shape())
    predict_request.add(affs, total_output_roi.get_shape())

    print("Starting prediction...")
    with build(pipeline):
        pipeline.request_batch(predict_request)
    print("Prediction finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("gunpowder.nodes.hdf5like_write_base").setLevel(logging.DEBUG)

    iteration = 190000

    raw_file = "../../../01_data/funke/zebrafinch/training/gt_z255-383_y1407-1663_x1535-1791.zarr"
    raw_dataset = "volumes/raw"
    auto_file = "../lsd/test_prediction.zarr"
    auto_dataset = "pred_lsds"
    out_file = "test_prediction.zarr"
    out_dataset = "pred_affs"

    predict(
        iteration, raw_file, raw_dataset, auto_file, auto_dataset, out_file, out_dataset
    )
