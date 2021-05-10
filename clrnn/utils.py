import torch
import copy
import os
import re
from avalanche.training.strategies import Replay, LwF, EWC, Naive, JointTraining, StreamingLDA
from shutil import copyfile
from pandas import read_csv
import numpy as np
import yaml
from types import SimpleNamespace
from sklearn.model_selection import ParameterGrid


def set_gpus(num_gpus):
    try:
        import gpustat
    except ImportError:
        print("gpustat module is not installed. No GPU allocated.")

    try:
        selected = []

        stats = gpustat.GPUStatCollection.new_query()

        for i in range(num_gpus):

            ids_mem = [res for res in map(lambda gpu: (int(gpu.entry['index']),
                                          float(gpu.entry['memory.used']) /\
                                          float(gpu.entry['memory.total'])),
                                      stats) if str(res[0]) not in selected]

            if len(ids_mem) == 0:
                # No more gpus available
                break

            best = min(ids_mem, key=lambda x: x[1])
            bestGPU, bestMem = best[0], best[1]
            # print(f"{i}-th best GPU is {bestGPU} with mem {bestMem}")
            selected.append(str(bestGPU))

        print("Setting GPUs to: {}".format(",".join(selected)))
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(selected)
    except BaseException as e:
        print("GPU not available: " + str(e))


def parse_config(config_file):
    """
    Parse yaml file containing also math notation like 1e-4
    """
    # fix to enable scientific notation
    # https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(config_file, 'r') as f:
        configs = yaml.load(f, Loader=loader)
    configs['config_file'] = config_file
    args = SimpleNamespace()
    for k, v in configs.items():
        args.__dict__[k] = v

    return args

def write_config_file(args, result_folder):
    """
    Write yaml configuration file inside result folder
    """
    with open(os.path.join(result_folder, 'config_file.yaml'), 'w') as f:
        yaml.dump(dict(vars(args)), f)

def create_result_folder(result_folder, path_save_models='saved_models'):
    '''
    Set plot folder by creating it if it does not exist.
    '''

    result_folder = os.path.expanduser(result_folder)
    os.makedirs(os.path.join(result_folder, path_save_models), exist_ok=True)
    return result_folder


def get_device(cuda):
    '''
    Choose device: cpu or cuda
    '''

    mode = 'cpu'
    if cuda and torch.cuda.is_available():
        mode = 'cuda'
    device = torch.device(mode)

    return device


def save_model(model, modelname, base_folder, path_save_models='saved_models', version=''):
    '''
    :param version: specify version of the model.
    Usually used to represent the model when trained after step 'version'
    '''

    torch.save(model.state_dict(), os.path.join(
        os.path.expanduser(base_folder),
        path_save_models, modelname+version+'.pt'))


def load_model(model, modelname, device, base_folder, path_save_models='saved_models', version=''):
    check = torch.load(os.path.join(
        os.path.expanduser(base_folder),
        path_save_models, modelname+version+'.pt'), map_location=device)

    model.load_state_dict(check)

    model.eval()

    return model


def create_grid(args, grid_arg='grid'):
    """
    Create grid search by returning a list of args.

    :parameter args: argument parser result
    :parameter grid_arg: field of `args` which contains
        a dictionary of
        'parameter name': list of possible values

    :return: list of configurations, one for each
        element in the grid search
    """

    try:
        grid = ParameterGrid(getattr(args, grid_arg))
    except AttributeError:
        print("Running without grid search")
        return [args]

    final_grid = []
    for el in grid:
        conf = copy.deepcopy(args)
        for k, v in el.items():
            conf.__dict__[k] = v
        final_grid.append(conf)

    return final_grid


def compute_average_eval_accuracy(folder,
                                  eval_result_name='eval_results.csv'):
    """
    Return average and std accuracy over all experiences
    after training on all experiences.
    """

    cur_file = os.path.join(folder, eval_result_name)
    data = read_csv(cur_file)
    data = data[data['training_exp'] == data['training_exp'].max()]  # choose last task
    data = data['eval_accuracy'].values

    acc = np.average(data, axis=0)
    acc_std = np.std(data, axis=0)

    return acc, acc_std

def compute_average_training_accuracy(folder,
                                      training_result_name='training_results.csv'):
    """
    Return average and std accuracy over all experiences
    after the last training epoch.
    """

    cur_file = os.path.join(folder, training_result_name)
    data = read_csv(cur_file)
    # select last epoch
    data = data[data['epoch'] == data['epoch'].max()]
    data = data['val_accuracy'].values

    # both are array of 2 elements (loss, acc)
    acc = np.average(data, axis=0)
    acc_std = np.std(data, axis=0)

    return acc, acc_std


def get_best_config(result_folder,
                    val_folder_name='VAL',
                    config_filename='config_file.yaml'):
    """
    Choose best config from a specific result folder containing
    model selection results. It produces a `best_config.yaml`
    file in the project root folder.

    :return: parsed args from the best configuration
    """

    best_config_filename = 'best_config.yaml'

    # find folders with format 'VAL{number}'
    ids = [str(el) for el in range(10)]
    dirs = [el for el in os.listdir(result_folder)
            if os.path.isdir(os.path.join(result_folder, el))
            and el.startswith(val_folder_name)
            and el[-1] in ids]

    best_dir = None
    best_acc = 0
    for dir_path in dirs:
        acc, _ = compute_average_eval_accuracy(os.path.join(result_folder, dir_path))
        if acc > best_acc:
            best_dir = dir_path
            best_acc = acc
    assert best_dir is not None, "Error in retrieving best accuracy"

    copyfile(os.path.join(result_folder, best_dir, config_filename),
             os.path.join(result_folder, best_config_filename))

    best_config = parse_config(os.path.join(result_folder, best_config_filename))

    return best_config


def get_strategy(model, optimizer, criterion, eval_plugin, device, args):
    if args.strategy == 'replay':
        cl_strategy = Replay(
            model, optimizer, criterion, train_mb_size=args.batch_size,
            train_epochs=args.epochs, device=device,
            evaluator=eval_plugin, eval_every=1,
            mem_size=args.mem_size
            )
    elif args.strategy == 'lwf':
        cl_strategy = LwF(
            model, optimizer, criterion, train_mb_size=args.batch_size,
            train_epochs=args.epochs, device=device,
            evaluator=eval_plugin, eval_every=1,
            temperature=args.lwf_temperature, alpha=args.lwf_alpha
            )
    elif args.strategy == 'ewc':
        cl_strategy = EWC(
            model, optimizer, criterion, train_mb_size=args.batch_size,
            train_epochs=args.epochs, device=device,
            evaluator=eval_plugin, eval_every=1,
            ewc_lambda=args.ewc_lambda, mode=args.ewc_mode
            )
    elif args.strategy == 'naive':
        cl_strategy = Naive(
            model, optimizer, criterion, train_mb_size=args.batch_size,
            train_epochs=args.epochs, device=device,
            evaluator=eval_plugin, eval_every=1
            )
    elif args.strategy == 'slda':
        cl_strategy = StreamingLDA(model, criterion, shrinkage_param=args.shrinkage,
            input_size=args.input_size, num_classes=args.num_classes,
            train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
            train_epochs=args.epochs, device=device,
            evaluator=eval_plugin, eval_every=1
        )
    elif args.strategy == 'joint':
        cl_strategy = JointTraining(
            model, optimizer, criterion, train_mb_size=args.batch_size,
            train_epochs=args.epochs, device=device,
            evaluator=eval_plugin
            )
    else:
        raise ValueError("Strategy name not recognized")

    return cl_strategy


__all__ = [
    'set_gpus',
    'parse_config',
    'get_device',
    'write_config_file',
    'create_result_folder',
    'save_model',
    'load_model',
    'create_grid',
    'compute_average_eval_accuracy',
    'compute_average_training_accuracy',
    'get_best_config',
    'get_strategy'
]
