import math
import multiprocessing
from pathlib import Path
import torch

from config.ConfigBase import Config


class ExperimentServer:
    def __init__(self):
        self.config = Config(Path.cwd() / 'config' / 'yaml' / 'server.yaml')

    def run(self, args):
        module = __import__('config')
        experiment = getattr(module, args.config.rstrip())(args.num_threads, args.device, args.shift, args.load)

        if args.inference:
            thread_params = experiment, 0, args.device, args.gpus[0]
            self.run_inference(thread_params)
        elif args.analysis != 'none':
            thread_params = experiment, args.analysis, args.device, args.gpus[0]
            self.run_analysis(thread_params)
        else:
            if self.config.backend == 'torch':
                self.run_training_torch_parallel(experiment, args.trials, args.device, args.gpus, args.trials_per_gpu)
            if self.config.backend == 'ray':
                print('running ray')

    def run_training_torch_parallel(self, experiment, trials, device, gpus, trials_per_gpu):
        multiprocessing.set_start_method('spawn')

        # for CPU
        gpu = -1
        total_gpus = 0
        total_segments = 1
        runs_per_segment = trials

        if device == 'cuda':
            total_gpus = len(gpus)
            if torch.cuda.device_count() < total_gpus:
                print("Warning: detected less devices than set in arguments, the number will be adjusted to available devices")
                total_gpus = torch.cuda.device_count()
            runs_per_segment = total_gpus * trials_per_gpu

            if trials > runs_per_segment:
                total_segments = math.ceil(trials / runs_per_segment)
            else:
                runs_per_segment = trials

        for n in range(total_segments):
            thread_params = []
            for i in range(n * runs_per_segment, min((n + 1) * runs_per_segment, trials)):
                if total_gpus > 0:
                    gpu = gpus[i % total_gpus]
                thread_params.append((experiment, i, device, gpu))

                print('Starting experiment No.{0}'.format(i))
                print(experiment)

            with multiprocessing.Pool(trials) as p:
                p.map(self.run_training, thread_params)

    @staticmethod
    def run_training(thread_params):
        experiment, i, device, gpu = thread_params

        if device == 'cuda':
            torch.cuda.set_device(gpu)

        experiment.train(i)

    @staticmethod
    def run_inference(thread_params):
        experiment, i, device, gpu = thread_params

        if device == 'cuda':
            torch.cuda.set_device(gpu)

        experiment.inference(i)

    @staticmethod
    def run_analysis(thread_params):
        experiment, task, device, gpu = thread_params

        if device == 'cuda':
            torch.cuda.set_device(gpu)

        experiment.analysis(task)

