import argparse
import neuroglancer
import numpy as np
import os
import sys
import zarr

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data-dir",
    "-d",
    type=str,
    action="append",
    help="The path to the zarr container to show",
)

parser.add_argument(
    "--bind-address",
    "-b",
    type=str,
    default="localhost",
    help="The bind address to use",
)

args = parser.parse_args()

neuroglancer.set_server_bind_address(args.bind_address)

f = zarr.open(args.data_dir[0])

datasets = ["volumes/raw", "volumes/labels/neuron_ids", "volumes/labels/labels_mask"]

viewer = neuroglancer.Viewer()

with viewer.txn() as s:
    for ds in datasets:
        res = f[ds].attrs["resolution"]
        offset = f[ds].attrs["offset"]

        dims = neuroglancer.CoordinateSpace(
            names=["z", "y", "x"], units="nm", scales=res
        )

        data = f[ds][:]

        if "mask" in ds:
            data *= 255

        layer = neuroglancer.LocalVolume(
            data=data, voxel_offset=offset, dimensions=dims
        )

        layer_type = (
            neuroglancer.SegmentationLayer
            if data.dtype == np.uint64
            else neuroglancer.ImageLayer
        )

        s.layers[ds] = layer_type(source=layer)

print(viewer)
