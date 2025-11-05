import argparse
import math
import os
import platform

import psutil
import torch

from experiment.ExperimentServer import ExperimentServer

if __name__ == '__main__':
    print(platform.system())
    print(torch.__version__)
    print(torch.__config__.show())
    print(torch.__config__.parallel_info())
    # torch.autograd.set_detect_anomaly(True)

    for i in range(torch.cuda.device_count()):
        print('{0:d}. {1:s}'.format(i, torch.cuda.get_device_name(i)))

    parser = argparse.ArgumentParser(description='Motivation models learning platform.')

    if not os.path.exists('./models'):
        os.mkdir('./models')

    parser.add_argument('--config', type=str, help='id of config')
    parser.add_argument('--device', type=str, help='device type', default='cpu')
    parser.add_argument('--gpus', help='device ids', default=None, nargs="+", type=int)
    parser.add_argument('--load', type=str, help='path to saved agent', default='')
    parser.add_argument('--inference', action="store_true", help='inference mode')
    parser.add_argument('--analysis', type=str, help='analytic task', default='none')
    parser.add_argument('-t', '--trials', type=int, help='total number of trials', default=1)
    parser.add_argument('-s', '--shift', type=int, help='shift result id', default=0)
    parser.add_argument('--trials_per_gpu', type=int, help='number of trials per GPU', default=1)
    parser.add_argument('--num_threads', type=int, help='number of parallel threads running in PPO (0=automatic number of cpus)', default=4)

    args = parser.parse_args()

    server = ExperimentServer()
    server.run(args)
