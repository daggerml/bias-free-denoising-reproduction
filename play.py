#!/usr/bin/env python3
import daggerml as dml
import executor_s3 as s3  # noqa: F401
from pathlib import Path
from pprint import pprint


IMAGE_ROOT = Path(__file__).parent / 'submodules/bias_free_denoising/data'


if __name__ == '__main__':
    with dml.Dag.new('bfdn.v0.dev') as dag:
        print('dag executor:', dag.executor)
        print()
        data = dag.load('bfdn.v0.data')
        print('data ==', data)
        print()
        print('data["test"] ==', data['test'])
        print()
        pprint(data.to_py())
        print()
        print()
        print('k8s:', dag.load('com.daggerml.resource.k8s').to_py())
        print()
        print('docker-build:', dag.load('com.daggerml.resource.docker').to_py())
