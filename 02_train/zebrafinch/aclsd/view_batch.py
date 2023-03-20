import argparse
import neuroglancer
import numpy as np
import os
import sys
import h5py

parser = argparse.ArgumentParser()

parser.add_argument(
    "--file", "-f", type=str, action="append", help="The path to the hdf batch to show"
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

f = h5py.File(args.file[0])

datasets = [
    "volumes/raw",
    "volumes/pretrained_lsd",
    "volumes/gt_affinities",
    "volumes/pred_affinities",
]


viewer = neuroglancer.Viewer()

shader = """
void main() {
    emitRGB(
        vec3(
            toNormalized(getDataValue(0)),
            toNormalized(getDataValue(1)),
            toNormalized(getDataValue(2)))
        );
}"""

with viewer.txn() as s:
    for ds in datasets:
        res = list(f[ds].attrs["resolution"])
        offset = [i / j for i, j in zip(list(f[ds].attrs["offset"]), res)]

        data = f[ds][:]

        if len(data.shape) == 4:
            names = ["c^", "x", "y", "z"]
            offset = [0] + offset
            res = [1] + res
        else:
            names = ["x", "y", "z"]

        dims = neuroglancer.CoordinateSpace(names=names, units="nm", scales=res)

        if "gt" in ds:
            data = data.astype(np.float32)

        layer = neuroglancer.LocalVolume(
            data=data, voxel_offset=offset, dimensions=dims
        )

        layer_type = (
            neuroglancer.SegmentationLayer
            if data.dtype == np.uint64
            else neuroglancer.ImageLayer
        )

        s.layers[ds] = layer_type(source=layer, shader=shader)

print(viewer)
