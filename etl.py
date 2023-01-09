#!/usr/bin/env python3
import boto3
import daggerml as dml
import executor_s3 as s3
import os
from pathlib import Path
from urllib.parse import urlparse


IMAGE_ROOT = Path(__file__).parent / 'submodules/bias_free_denoising/data'


if __name__ == '__main__':
    assert 'DML_BUCKET' in os.environ, 'Please set DML_BUCKET'
    with dml.Dag.new('bfdn.v0.data') as dag:
        s3c = boto3.client('s3')
        out = {'test': {}}
        # TRAIN
        print('uploading train set')
        out['train'] = s3_node = s3.new_prefix(dag)
        uri = urlparse(s3_node.to_py().uri)
        bucket, prefix = uri.netloc, uri.path.strip('/')
        for path in (IMAGE_ROOT / 'Train400/').glob('*.png'):
            s3c.upload_file(path, bucket, f'{prefix}/{path.name}')
        for n in ['Kodak23', 'Set12', 'Set68']:
            print('uploading test set:', n)
            out['test'][n] = s3_node = s3.new_prefix(dag)
            uri = urlparse(s3_node.to_py().uri)
            bucket, prefix = uri.netloc, uri.path.strip('/')
            for path in (IMAGE_ROOT / 'Test' / n).glob('*.png'):
                s3c.upload_file(path, bucket, f'{prefix}/{path.name}')
        dag.commit(out)
