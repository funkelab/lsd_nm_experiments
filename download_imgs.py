import boto3
import os

if __name__ == "__main__":
    # set bucket credentials
    access_key = ""
    secret_key = ""
    bucket = ""

    client = boto3.client(
        "s3", aws_access_key_id=access_key, aws_secret_access_key=secret_key
    )

    files = [
        "lsd:v0.8.img",
        # "lsd_legacy.img",
    ]

    for f in files:
        bucket_key = f"funke/singularity/{f}"

        print(f"downloading {f} from {bucket_key} \n")

        client.download_file(Bucket=bucket, Key=bucket_key, Filename=f)
