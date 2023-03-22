import boto3
import os
from botocore import UNSIGNED
from botocore.config import Config

if __name__ == "__main__":
    bucket = "open-neurodata"

    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    files = [
        "lsd:v0.8.img",
        # "lsd_legacy.img",
    ]

    for f in files:
        bucket_key = f"funke/singularity/{f}"

        print(f"downloading {f} from {bucket_key} \n")

        client.download_file(Bucket=bucket, Key=bucket_key, Filename=f)
