import argparse
import math
import os
import platform

import psutil
import ray
import torch

import PPO_HardAtariGame
import PPO_ProcgenGame
from experiment.ExperimentServer import ExperimentServer

envs = {
    'ppo': {
        'gravitar': {'name': 'GravitarNoFrameskip-v4', 'class': PPO_HardAtariGame},
        'montezuma': {'name': 'MontezumaRevengeNoFrameskip-v4', 'class': PPO_HardAtariGame},
        'pitfall': {'name': 'PitfallNoFrameskip-v4', 'class': PPO_HardAtariGame},
        'private_eye': {'name': 'PrivateEyeNoFrameskip-v4', 'class': PPO_HardAtariGame},
        'solaris': {'name': 'SolarisNoFrameskip-v4', 'class': PPO_HardAtariGame},
        'venture': {'name': 'VentureNoFrameskip-v4', 'class': PPO_HardAtariGame},
        'adventure': {'name': 'Adventure-v4', 'class': PPO_HardAtariGame},
        'caveflyer': {'name': 'procgen-caveflyer-v0', 'class': PPO_ProcgenGame},
        'coinrun': {'name': 'procgen-coinrun-v0', 'class': PPO_ProcgenGame},
        'climber': {'name': 'procgen-climber-v0', 'class': PPO_ProcgenGame},
        'jumper': {'name': 'procgen-jumper-v0', 'class': PPO_ProcgenGame},
    },
}


def run_ray_parallel(args, experiment):
    @ray.remote(num_gpus=1 / args.num_processes, max_calls=1)
    def run_thread_ray(p_thread_params):
        run_thread(p_thread_params)

    for i in range(math.ceil(experiment.trials / args.num_processes)):
        thread_params = []
        for j in range(args.num_processes):
            index = i * args.num_processes + j
            if index < experiment.trials:
                thread_params.append((args.algorithm, args.env, experiment, index))

        ray.get([run_thread_ray.remote(tp) for tp in thread_params])


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
    parser.add_argument('-t', '--trials', type=int, help='total number of trials', default=1)
    parser.add_argument('-s', '--shift', type=int, help='shift result id', default=0)
    parser.add_argument('--trials_per_gpu', type=int, help='number of trials per GPU', default=1)
    parser.add_argument('--num_threads', type=int, help='number of parallel threads running in PPO (0=automatic number of cpus)', default=4)

    args = parser.parse_args()

    server = ExperimentServer()
    server.run(args)

    # if args.load != '':
    #     env_class = envs[args.algorithm][args.env]
    #     env_class.test(experiment, args.load)
    # else:
    #     if args.thread:
    #         experiment.trials = 1
    #
    #     if args.parallel:
    #         num_cpus = psutil.cpu_count(logical=True)
    #         print('Running parallel {0} trainings'.format(experiment.trials))
    #         print('Using {0} parallel backend'.format(args.parallel_backend))
    #
    #         if args.parallel_backend == 'ray':
    #             if args.gpus:
    #                 experiment.gpus = None
    #                 os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])
    #             ray.shutdown()
    #             ray.init(num_cpus=num_cpus, num_gpus=1)
    #             torch.set_num_threads(max(1, num_cpus // experiment.trials))
    #
    #             run_ray_parallel(args, experiment)
    #             # write_command_file(args, experiment)
    #             # run_command_file()
    #         elif args.parallel_backend == 'torch':
    #             torch.set_num_threads(1)
    #             run_torch_parallel(args, experiment)
    #     else:
    #         for i in range(experiment.trials):
    #             run(i, args.algorithm, args.env, experiment)
