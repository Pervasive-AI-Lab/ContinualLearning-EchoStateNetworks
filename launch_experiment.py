import argparse
import copy
import os
#os.environ["OMP_NUM_THREADS"] = "1"  # This is CRUCIAL to avoid bottlenecks when running experiments in parallel. DO NOT REMOVE IT
import ray
import torch
from clrnn.utils import set_gpus, parse_config, create_result_folder, \
    create_grid, get_best_config, write_config_file
from experiments.splitmnist_esn import smnist_esn
from experiments.ssc_esn import ssc_esn
from experiments.joint_train import joint_smnist, joint_ssc

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='', help='path to yaml configuration file')
args = parser.parse_args()

if args.config_file == '':
    raise ValueError('You must provide a config file.')

args = parse_config(args.config_file)
torch.set_num_threads(args.max_cpus)
orig_args = copy.deepcopy(args)
grid_args = create_grid(args)


def run_exp_noray(argum):
    create_result_folder(argum.result_folder)
    if argum.exp_type == 'smnist_esn':
        smnist_esn(argum)
    elif argum.exp_type == 'ssc_esn':
        ssc_esn(argum)
    elif argum.exp_type == 'joint_ssc':
        joint_ssc(argum)
    elif argum.exp_type == 'joint_smnist':
        joint_smnist(argum)
    else:
        print("Experiment type not recognized.")


@ray.remote(num_cpus=args.cpus_per_job, num_gpus=args.gpus_per_job)
def run_exp(argum):
    if args.cuda:
        print(f'Using GPU {ray.get_gpu_ids()}')
    else:
        print('Using CPUs')

    if argum.exp_type == 'smnist_esn':
        smnist_esn(argum)
    elif argum.exp_type == 'ssc_esn':
        ssc_esn(argum)
    elif argum.exp_type == 'joint_ssc':
        joint_ssc(argum)
    elif argum.exp_type == 'joint_smnist':
        joint_smnist(argum)
    else:
        print("Experiment type not recognized.")


if args.cuda:
    set_gpus(args.max_gpus)

if args.no_ray:
    run_exp_noray(args)
else:
    try:
        if args.cuda:
            # Execution will be sequential
            ray.init(num_cpus=args.max_cpus, num_gpus=args.max_gpus)
        elif os.environ.get('ip_head') is not None:
            assert os.environ.get('redis_password') is not None, "Missing redis password"
            ray.init(address=os.environ.get('ip_head'), _redis_password=os.environ.get('redis_password'))
            print("Connected to Ray cluster.")
            print(f"Available nodes: {ray.nodes()}")
            args.gpus_per_job = 0
        else:
            ray.init(num_cpus=args.max_cpus)
            args.gpus_per_job = 0
            print(f"Started local ray instance.")

        assert ray.is_initialized(), "Error in initializing ray."

        if len(grid_args) > 1:
            #######
            ####### START MODEL SELECTION WITH GRID SEARCH
            #######
            remaining_ids = []
            for grid_id, curr_args in enumerate(grid_args):
                # create jobs
                curr_args.result_folder = os.path.join(orig_args.result_folder, f'VAL{grid_id}')
                create_result_folder(curr_args.result_folder)
                write_config_file(curr_args, curr_args.result_folder)
                curr_args.model_selection = True
                remaining_ids.append(run_exp.remote(curr_args))
            n_jobs = len(remaining_ids)
            print(f"Scheduled jobs: {n_jobs}")

            # wait for jobs
            while remaining_ids:
                done_ids, remaining_ids = ray.wait(remaining_ids, num_returns=1)
                for result_id in done_ids:
                    # There is only one return result by default.
                    result = ray.get(result_id)
                    n_jobs -= 1
                    print(f'Job {result_id} terminated. Jobs left: {n_jobs}')

            best_args = get_best_config(orig_args.result_folder)
        else:
            best_args = copy.deepcopy(orig_args)

        ######
        ###### START ASSESSMENT
        ######
        remaining_ids = []
        for i in range(best_args.assess_runs):
            best_args.result_folder = os.path.join(orig_args.result_folder, f'ASSESS{i}')
            create_result_folder(best_args.result_folder)
            write_config_file(best_args, best_args.result_folder)
            best_args.model_selection = False
            remaining_ids.append(run_exp.remote(best_args))
        n_jobs = len(remaining_ids)
        print(f"Scheduled jobs: {n_jobs}")

        # wait for jobs
        while remaining_ids:
            done_ids, remaining_ids = ray.wait(remaining_ids, num_returns=1)
            for result_id in done_ids:
                # There is only one return result by default.
                result = ray.get(result_id)
                n_jobs -= 1
                print(f'Job {result_id} terminated. Jobs left: {n_jobs}')

    finally:
        print('Shutting down ray...')
        ray.shutdown()
        print('Ray closed.')
