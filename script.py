#!/usr/bin/env python3
import argparse
from bfdn.train import main
from bfdn.util import DATA_PATH


def get_args():
    parser = argparse.ArgumentParser(
        description="bfdn",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--num-layers", type=int, default=17,
                        help="Number of total layers")
    parser.add_argument("--num-epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--milestone", type=int, default=30,
                        help="When to decay learning rate")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--use-bias", action='store_true',
                        help='include bias?')
    parser.add_argument("--out-loc", type=str, default=f'{DATA_PATH}/results',
                        help='path of log files')
    parser.add_argument("--noise-low", type=float, default=0.1,
                        help='lower bound on noise level')
    parser.add_argument("--noise-high", type=float, default=55,
                        help='upper bound on noise level')
    parser.add_argument("--valid-noise", type=float, default=35,
                        help='noise level for validation')
    parser.add_argument("--extra-images", type=str, default=[], nargs='*',
                        help='noise level for validation')
    parser.add_argument("--debug", action='store_true',
                        help='run with fewer training images')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    exit(main(**vars(args)))
