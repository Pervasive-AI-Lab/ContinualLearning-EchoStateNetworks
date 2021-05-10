#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from torch.utils.data import random_split
from torch.nn import CrossEntropyLoss, LSTM
from torch.optim import Adam, SGD

from avalanche.benchmarks.generators import nc_benchmark
from clrnn import DeepReservoirClassifier, SequenceClassifier
from clrnn.benchmarks.seq_mnist import SequentialMNIST
from clrnn.benchmarks import SSC
from avalanche.evaluation.metrics import accuracy_metrics, \
    loss_metrics, timing_metrics, forgetting_metrics
from avalanche.logging import TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from clrnn.utils import get_device, save_model
from avalanche.logging import CSVLogger
from clrnn.utils import get_strategy


def joint_smnist(args):

    # --- SCENARIO (split seqMNIST)
    pix = args.pixels
    dataset_train = SequentialMNIST(args.dataroot, train=True, pixel_per_step=pix)
    train_length = int(len(dataset_train)*0.75)
    val_length = len(dataset_train) - train_length
    dataset_train, dataset_val = random_split(dataset_train, [train_length, val_length])
    dataset_test = SequentialMNIST(args.dataroot, train=False, pixel_per_step=pix)
    if args.model_selection:
        scenario = nc_benchmark(dataset_train, dataset_val, 10,
                               task_labels=False, seed=1234)
    else:
        scenario = nc_benchmark(dataset_train, dataset_test, 10,
                               task_labels=False, seed=1234)

    run_joint(args, pix, scenario)


def joint_ssc(args):
    input_size = 40

    dataset_train = SSC(args.dataroot, split='train', n_mels=input_size, debug=False)
    train_length = int(len(dataset_train)*0.75)
    val_length = len(dataset_train) - train_length
    dataset_train, dataset_val = random_split(dataset_train, [train_length, val_length])
    dataset_test = SSC(args.dataroot, split='test', n_mels=input_size, debug=False)
    if args.model_selection:
        scenario = nc_benchmark(dataset_train, dataset_val, 10,
                               task_labels=False, seed=1234)
    else:
        scenario = nc_benchmark(dataset_train, dataset_test, 10,
                               task_labels=False, seed=1234)

    run_joint(args, input_size, scenario)


def run_joint(args, input_size, scenario):

    device = get_device(args.cuda)

    ## ---  MODEL
    model = DeepReservoirClassifier(input_size,
                                    scenario.n_classes,
                                    units=args.esn_units,
                                    layers=args.esn_layers,
                                    spectral_radius=args.spectral_radius,
                                    input_scaling=args.input_scaling,
                                    feedforward_layers=1,
                                    connectivity_input=args.conn_inp,
                                    connectivity_recurrent=args.conn_rec,
                                    leaky=args.leaky
                                    )

    criterion = CrossEntropyLoss()
    if args.opt_name == 'adam':
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
    elif args.opt_name == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError("Optimizer name not recognized.")


    # --- STRATEGY
    f = open(os.path.join(args.result_folder, 'text_logger.txt'), 'w')
    text_logger = TextLogger(f)
    csv_logger = CSVLogger(args.result_folder)
    tensorboard_logger = TensorboardLogger(os.path.join(args.result_folder, "tb_data"))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[text_logger, csv_logger, tensorboard_logger])

    args.strategy = 'joint'
    cl_strategy = get_strategy(model, optimizer, criterion, eval_plugin, device, args)

    # --- TRAINING LOOP
    print('Starting experiment...')
    save_model(model, 'esn', args.result_folder, version='_init')

    cl_strategy.train(scenario.train_stream)
    cl_strategy.eval(scenario.test_stream)

    f.close()
    csv_logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=bool, action="store_true", help='Use CPU')
    args = parser.parse_args()

    joint_smnist(args)

