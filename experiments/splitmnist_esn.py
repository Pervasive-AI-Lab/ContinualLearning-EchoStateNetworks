#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from torch.nn import CrossEntropyLoss, LSTM
from torch.optim import Adam, SGD

from avalanche.benchmarks.generators import nc_benchmark
from clrnn import DeepReservoirClassifier, SequenceClassifier, ESNWrapper
from clrnn.benchmarks.seq_mnist import SequentialMNIST
from avalanche.evaluation.metrics import accuracy_metrics, \
    loss_metrics, timing_metrics, forgetting_metrics
from avalanche.logging import TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from clrnn.utils import get_device, save_model
from avalanche.logging import CSVLogger
from clrnn.utils import get_strategy


def smnist_esn(args):

    device = get_device(args.cuda)

    # --- SCENARIO (split seqMNIST)
    pix = args.pixels
    mnist_train = SequentialMNIST(args.dataroot, train=True, pixel_per_step=pix)
    mnist_test = SequentialMNIST(args.dataroot, train=False, pixel_per_step=pix)
    scenario = nc_benchmark(mnist_train, mnist_test, 5,
                           task_labels=False, seed=1234,
                           fixed_class_order=list(range(10)))

    if 'use_rnn' not in vars(args):
        args.use_rnn = False

    ## ---  MODEL
    if 'leaky' not in vars(args):
        args.leaky = 1

    if not args.use_rnn:
        model = DeepReservoirClassifier(pix,
                                        scenario.n_classes,
                                        units=args.esn_units,
                                        layers=args.esn_layers,
                                        spectral_radius=args.spectral_radius,
                                        input_scaling=args.input_scaling,
                                        feedforward_layers=1,
                                        leaky=args.leaky
                                        )
        if args.strategy == 'slda':
            model = ESNWrapper(model, 'hidden')
    else:
        model = SequenceClassifier(
            LSTM(pix, args.rnn_units, batch_first=True),
            args.rnn_units, scenario.n_classes)

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

    # same number of classes per exp
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[text_logger, csv_logger, tensorboard_logger])

    cl_strategy = get_strategy(model, optimizer, criterion, eval_plugin, device, args)

    # --- TRAINING LOOP
    print('Starting experiment...')
    save_model(model, 'esn', args.result_folder, version='_init')

    for i, exp in enumerate(scenario.train_stream):
        print("Start training on experience ", exp.current_experience)
        print("Classes in experience: ", exp.classes_in_this_experience)

        cl_strategy.train(exp, eval_streams=[scenario.test_stream[i]])
        print('Training completed')
        if args.strategy == 'slda':
            cl_strategy.save_model(os.path.join(args.result_folder, 'saved_models'), f'esn{i}')
        else:
            save_model(model, 'esn', args.result_folder, version=str(i))
        print('Computing accuracy on the whole test set')
        cl_strategy.eval(scenario.test_stream)


    f.close()
    csv_logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=bool, action="store_true", help='Use CPU')
    args = parser.parse_args()

    smnist_esn(args)

