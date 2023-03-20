import boto3
import os


if __name__ == "__main__":
    # set bucket credentials
    access_key = "AKIA4XXGEV6ZQOTMTHX6"
    secret_key = "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"
    bucket_name = "open-neurodata"

    resource = boto3.resource(
        "s3", aws_access_key_id=access_key, aws_secret_access_key=secret_key
    )

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
