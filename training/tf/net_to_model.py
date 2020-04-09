#!/usr/bin/env python3
import os
import sys
import argparse
from tfprocess import TFProcess

from config import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("network",
        help='Training network filename', nargs='?', type=str)
    parser.add_argument("--logbase", default='leelalogs', type=str,
        help="Log file prefix (for tensorboard) (default: %(default)s)")
    parser.add_argument("--sample", default=DOWN_SAMPLE, type=int,
        help="Rate of data down-sampling to use (default: %(default)d)")
    parser.add_argument("--bufferbits", default=TRAIN_SHUFFLE_BITS, type=int,
        help="Train shuffle-buffer size in bits (default: %(default)d)")
    parser.add_argument("--rate", default=LEARN_RATE, type=float,
        help="Learning rate (default: %(default)f)")
    parser.add_argument("--minsteps", default=FIRST_NETWORK_STEPS , type=int,
        help="First network after this many steps (default: %(default)d)")
    parser.add_argument("--steps", default=TRAINING_STEPS, type=int,
        help="Training step before writing a network (default: %(default)d)")
    parser.add_argument("--maxsteps", default=MAX_TRAINING_STEPS, type=int,
        help="Terminates after this many steps (default: %(default)d)")
    parser.add_argument("--maxkeep", default=MAX_SAVER_TO_KEEP, type=int,
        help="Keeps meta files for at most this many networks (default: %(default)d)")
    parser.add_argument("--policyloss", default=POLICY_LOSS_WT, type=float,
        help="Coefficient for policy term in loss function (default: %(default)f)")
    parser.add_argument("--mseloss", default=MSE_LOSS_WT, type=float,
        help="Coefficient for mse term in loss function (default: %(default)f)")
    parser.add_argument("--regloss", default=REG_LOSS_WT, type=float,
        help="Coefficient for regularizing term in loss function (default: %(default)f)")
    args = parser.parse_args()

    with open(args.network, 'r') as f:
        weights = []
        for e, line in enumerate(f):
            if e == 0:
                #Version
                print("Version", line.strip())
                # if line != '1\n':
                #     raise ValueError("Unknown version {}".format(line.strip()))
            else:
                weights.append(list(map(float, line.split(' '))))
            if e == 2:
                channels = len(line.split(' '))
                print("Channels", channels)

        blocks = e - (4 + 14)
        if blocks % 8 != 0:
            raise ValueError("Inconsistent number of weights in the file")
        blocks //= 8
        print("Blocks", blocks)

    tfprocess = TFProcess(blocks, channels,
                        args.rate, args.minsteps, args.steps, args.maxsteps, args.maxkeep,
                        args.policyloss, args.mseloss, args.regloss)
    tfprocess.init(batch_size=1, gpus_num=1)
    tfprocess.replace_weights(weights)
    path = os.path.join(os.getcwd(), "leelaz-model")
    save_path = tfprocess.saver.save(tfprocess.session, path, global_step=0)

if __name__ == "__main__":
    main()
