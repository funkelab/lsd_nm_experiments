import boto3
import os
from botocore import UNSIGNED
from botocore.config import Config


if __name__ == "__main__":
    bucket_name = "open-neurodata"

    resource = boto3.resource("s3", config=Config(signature_version=UNSIGNED))

    bucket = resource.Bucket(bucket_name)

    base = os.path.abspath("../02_train")

    datasets = [
        "zebrafinch",
        # 'hemi',
        # 'fib25'
    ]

    networks = [
        "baseline",
        # 'mtlsd',
        # 'lsd',
        # 'long_range',
        # 'malis',
        # 'aclsd',
        # 'acrlsd'
    ]

    for dataset in datasets:
        for network in networks:
            out_dir = os.path.join(base, dataset, network)

            key = f"funke/{dataset}/training/checkpoints/{network}/"

            for obj in bucket.objects.filter(Prefix=key):
                down = obj.key
                out = os.path.join(out_dir, obj.key.replace(key, ""))

                bucket.download_file(down, out)
