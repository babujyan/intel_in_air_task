from pathlib import Path
import boto3
import botocore
from botocore import UNSIGNED
from botocore.config import Config
import os

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))


def get_file(file_path, dest_path, bucket="intelinair-internship"):
    if not Path(os.path.join(dest_path, file_path)).isfile():
        PATH, file = os.path.split(file_path)
        if not os.path.isdir(PATH):
            os.makedirs(PATH)
        s3.download_file(bucket, PATH, os.path.join(dest_path, PATH))
