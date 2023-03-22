import boto3
import json
import multiprocessing as mp
import os
from botocore import UNSIGNED
from botocore.config import Config

client = None


def initialize(access_key, secret_key):
    global client

    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))


# function to download all files nested in a bucket path
def download_data(job):
    bucket_name, path = job

    resource = boto3.resource("s3", config=Config(signature_version=UNSIGNED))

    bucket = resource.Bucket(bucket_name)

    for obj in bucket.objects.filter(Prefix=path):
        os.makedirs(os.path.dirname(obj.key), exist_ok=True)

        key = obj.key

        print(f"Downloading {key}")
        bucket.download_file(key, key)


if __name__ == "__main__":
    bucket = "open-neurodata"

    # load training data
    with open("datasets.json") as f:
        config = json.load(f)

    # just test with one volume for each dataset.
    volumes = {
        "fib25": config["fib25"][0:1],
        "hemi": config["hemi"][0:1],
        "zebrafinch": config["zebrafinch"][0:1],
    }

    jobs = [(bucket, f"funke/{d}/training/{x}") for d, v in volumes.items() for x in v]

    # download each volume with separate process, would want to adapt to work
    # with more processes if downloading more than 3 volumes..
    pool = mp.Pool(len(jobs), initialize(access_key, secret_key))

    pool.map(download_data, jobs)

    pool.close()
    pool.join()
