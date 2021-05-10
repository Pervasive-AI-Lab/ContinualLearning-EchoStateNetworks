from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.nn import LSTM
from avalanche.benchmarks.generators import nc_benchmark
from clrnn import SequenceClassifier, DeepReservoirClassifier, ESNWrapper
from avalanche.evaluation.metrics import accuracy_metrics, \
    loss_metrics, timing_metrics, forgetting_metrics
from avalanche.logging import TextLogger, TensorboardLogger
from avalanche.logging import CSVLogger
from avalanche.training.plugins import EvaluationPlugin

from clrnn.benchmarks import SSC
from clrnn.utils import get_device, get_strategy, save_model

def ssc_esn(args):

    device = get_device(args.cuda)

    # --- SCENARIO
    input_size = 40

    if 'model_selection' not in vars(args):
        args.model_selection = False

    if args.model_selection:
        classes = ['bed', 'bird', 'cat', 'dog', 'down', 'eight']
        n_exp = 3
    else:
        classes = ['house', 'left', 'marvel', 'nine', 'no', 'off', 'one', 'on', 'right', 'seven', 'sheila',
                   'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']
        n_exp = 10

    dataset_train = SSC(args.dataroot, split='train', n_mels=input_size, debug=False, classes=classes)
    dataset_test = SSC(args.dataroot, split='test', n_mels=input_size, debug=False, classes=classes)
    scenario = nc_benchmark(dataset_train, dataset_test, n_exp,
                            task_labels=False, seed=1234,
                            fixed_class_order=list(range(len(classes))))

    if 'use_rnn' not in vars(args):
        args.use_rnn = False

    ## ---  MODEL
    if not args.use_rnn:
        model = DeepReservoirClassifier(input_size,
                                        scenario.n_classes,
                                        units=args.esn_units,
                                        layers=args.esn_layers,
                                        spectral_radius=args.spectral_radius,
                                        input_scaling=args.input_scaling,
                                        feedforward_layers=args.feed_layers,
                                        connectivity_input=args.conn_inp,
                                        connectivity_recurrent=args.conn_rec,
                                        leaky=args.leaky)
        if args.strategy == 'slda':
            model = ESNWrapper(model, 'hidden')

    else:
        model = SequenceClassifier(
            LSTM(input_size, args.rnn_units, batch_first=True, num_layers=2),
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
    parser.add_argument('--dataroot', type=str, default='/data/cossu/speech_words')
    parser.add_argument('--logdir', type=str, default='/data/cossu/experiments/test')
    parser.add_argument('--hs', type=int, default=128, help='LSTM hidden size')
    args = parser.parse_args()

    ssc_esn(args)
