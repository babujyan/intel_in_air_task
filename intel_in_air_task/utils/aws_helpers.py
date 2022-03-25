from pathlib import Path
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import os

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))


def get_file(file_path, dest_path, bucket="intelinair-internship"):
    """
    Download file from s3 if it does not exist in filesystem
    :param file_path: file path in a bucket
    :param dest_path: destination directory for saving the file
    :param bucket: bucket name
    """
    if not Path(os.path.join(dest_path, file_path)).isfile():
        path, file = os.path.split(file_path)
        if not os.path.isdir(path):
            os.makedirs(path)
        s3.download_file(bucket, path, os.path.join(dest_path, path))
