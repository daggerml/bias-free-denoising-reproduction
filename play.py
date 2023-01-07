#!/usr/bin/env python3
import boto3
import daggerml as dml
import executor_s3 as s3
from pathlib import Path
from pprint import pprint
from urllib.parse import urlparse


IMAGE_ROOT = Path(__file__).parent / 'submodules/bias_free_denoising/data'


if __name__ == '__main__':
    with dml.Dag.new('bfdn.v0.dev') as dag:
        n = dag.load('bfdn.v0.data')
        print('n ==', n)
        print()
        print('n["test"] ==', n['test'])
        print()
        pprint(n.to_py())
        print(1/0)
