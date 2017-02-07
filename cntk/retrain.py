# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import argparse
import math
import numpy as np

from cntk.utils import *
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
from cntk import Trainer, cntk_py
from cntk import load_model
from cntk.learner import momentum_sgd, learning_rate_schedule, momentum_as_time_constant_schedule, UnitType
from _cntk_py import set_computation_network_trace_level

from resnet_models import *

# Paths relative to current python file.
data_path  = 'E:\\combined\\train_subsample2'
model_path = 'E:\\cntk\\models'

# model dimensions
image_height = 224
image_width  = 224
num_channels = 3  # RGB
num_classes  = 7

# Define the reader for both training and evaluation action.
def create_reader(map_file, mean_file, train):
    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        raise RuntimeError("File '{}' or '{}' does not exist".format(map_file, mean_file))
    print('trying to create the reader')

    # transformation pipeline for the features has jitter/crop only when training
    transforms = [ImageDeserializer.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')]
    # deserializer
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features = StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
        labels   = StreamDef(field='label', shape=num_classes))))   # and second as 'label'


# Train and evaluate the network.
def train(reader_train, network_name, epoch_size, max_epochs, model_location=None):


    set_computation_network_trace_level(0)

    # Input variables denoting the features and label data
    input_var = input_variable((num_channels, image_height, image_width))
    label_var = input_variable((num_classes))


    if network_name == 'resnet20': 
        z = create_model(input_var, 3, num_classes)
        lr_per_mb = [0.1]*80+[0.01]*40+[0.001]
    elif network_name == 'resnet50': 
        z = create_model(input_var, 8, num_classes)
        lr_per_mb = [0.1]*1+[1.0]*80+[0.1]*40+[0.01]
    elif network_name == 'resnet110': 
        z = create_model(input_var, 18, num_classes)
        lr_per_mb = [0.1]*1+[1.0]*80+[0.1]*40+[0.01]
    else: 
        return RuntimeError("Unknown model name!")
    print('Successfully created the Resnet')

    # loss and metric
    ce = cross_entropy_with_softmax(z, label_var)
    pe = classification_error(z, label_var)

    # shared training parameters 
    minibatch_size = 16
    momentum_time_constant = -minibatch_size/np.log(0.9)
    l2_reg_weight = 0.0001

    # Set learning parameters
    lr_per_sample = [lr/minibatch_size for lr in lr_per_mb]
    lr_schedule = learning_rate_schedule(lr_per_sample, epoch_size=epoch_size, unit=UnitType.sample)
    mm_schedule = momentum_as_time_constant_schedule(momentum_time_constant)
    
    # trainer object
    learner     = momentum_sgd(z.parameters, lr_schedule, mm_schedule,
                               l2_regularization_weight = l2_reg_weight)
    trainer     = Trainer(z, ce, pe, learner)
    if model_location is not None:
        trainer.restore_from_checkpoint(model_location)

    # define mapping from reader streams to network inputs
    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    log_number_of_parameters(z) ; print()
    progress_printer = ProgressPrinter(tag='Training')

    # perform model training
    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size-sample_count), input_map=input_map) # fetch minibatch.
            trainer.train_minibatch(data)                                   # update model with it
            sample_count += trainer.previous_minibatch_sample_count         # count samples processed so far
            progress_printer.update_with_trainer(trainer, with_metric=True) # log progress
        progress_printer.epoch_summary(with_metric=True)
        z.save_model(os.path.join(model_path, network_name + "_{}.dnn".format(epoch)))
    print('done training')
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', help='network type, resnet20, resnet50, or resnet110', required=False, default='resnet50')
    parser.add_argument('-e', '--epochs', help='total epochs', required=False, default='160')

    args = vars(parser.parse_args())
    epochs = int(args['epochs'])
    network_name = args['network']
    
    reader_train = create_reader(os.path.join(data_path, 'map.txt'), os.path.join(data_path, 'mean.txt'), True)
    print('Successfully created the reader')
    epoch_size = 10000
    train(reader_train, network_name, epoch_size, epochs)
