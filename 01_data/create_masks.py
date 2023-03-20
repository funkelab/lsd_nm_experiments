import glob
import numpy as np
import os
import zarr

volumes = ["fib25", "hemi", "zebrafinch"]

samples = [glob.glob(os.path.join(f"funke/{v}/training", "*.zarr")) for v in volumes]

samples = [i for s in samples for i in s]

labels_name = "volumes/labels/neuron_ids"
labels_mask_name = "volumes/labels/labels_mask"

for sample in samples:
    f = zarr.open(sample, "a")

    labels = f[labels_name][:]
    offset = f[labels_name].attrs["offset"]
    resolution = f[labels_name].attrs["resolution"]

    labels_mask = np.ones_like(labels).astype(np.uint8)

    if "hemi" in sample:
        background_mask = labels == np.uint64(-3)
        labels_mask[background_mask] = 0

    f[labels_mask_name] = labels_mask
    f[labels_mask_name].attrs["offset"] = offset
    f[labels_mask_name].attrs["resolution"] = resolution
