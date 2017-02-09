# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

''' running parameters -- edit as necessary '''
data_path  = 'E:\\combined\\train_subsample'
model_path = 'E:\\cntk\\models'
image_height = 224
image_width  = 224
num_channels = 3  # RGB
num_classes  = 7

import os
import math
import numpy as np

from cntk.initializer import he_normal
from cntk.layers import AveragePooling, BatchNormalization, Convolution, Dense
from cntk.utils import *
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error, element_times, relu
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
from cntk import Trainer, cntk_py
from cntk.learner import momentum_sgd, learning_rate_schedule, momentum_as_time_constant_schedule, UnitType
from _cntk_py import set_computation_network_trace_level

# Helper functions for ResNet construction
def conv_bn(input, filter_size, num_filters, strides=(1,1), init=he_normal()):
    c = Convolution(filter_size, num_filters, activation=None, init=init, pad=True, strides=strides, bias=False)(input)
    r = BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=False)(c)
    return r

def conv_bn_relu(input, filter_size, num_filters, strides=(1,1), init=he_normal()):
    r = conv_bn(input, filter_size, num_filters, strides, init) 
    return relu(r)

def resnet_basic(input, num_filters):
    c1 = conv_bn_relu(input, (3,3), num_filters)
    c2 = conv_bn(c1, (3,3), num_filters)
    p  = c2 + input
    return relu(p)

def resnet_basic_inc(input, num_filters, strides=(2,2)):
    c1 = conv_bn_relu(input, (3,3), num_filters, strides)
    c2 = conv_bn(c1, (3,3), num_filters)
    s  = conv_bn(input, (1,1), num_filters, strides)
    p  = c2 + s
    return relu(p)

def resnet_basic_stack(input, num_stack_layers, num_filters): 
    assert (num_stack_layers >= 0)
    l = input 
    for _ in range(num_stack_layers): 
        l = resnet_basic(l, num_filters)
    return l 

def create_model(input, num_stack_layers, num_classes):
    c_map = [16, 32, 64]
    conv = conv_bn_relu(input, (3,3), c_map[0])
    r1 = resnet_basic_stack(conv, num_stack_layers, c_map[0])
    r2_1 = resnet_basic_inc(r1, c_map[1])
    r2_2 = resnet_basic_stack(r2_1, num_stack_layers-1, c_map[1])
    r3_1 = resnet_basic_inc(r2_2, c_map[2])
    r3_2 = resnet_basic_stack(r3_1, num_stack_layers-1, c_map[2])
    pool = AveragePooling(filter_shape=(8,8))(r3_2) 
    z = Dense(num_classes)(pool)
    return z

# Function for accessing and preprocessing the images
def create_reader(map_file):
    if not os.path.exists(map_file):
        raise RuntimeError("File '{}' does not exist".format(map_file))

    transforms = [ImageDeserializer.crop(crop_type='randomarea', area_ratio=[0.85,1.0],
                                         jitter_type='uniratio'),
                  ImageDeserializer.scale(width=image_width,
                                          height=image_height,
                                          channels=num_channels,
                                          interpolations='linear')]
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features = StreamDef(field='image', transforms=transforms),
        labels   = StreamDef(field='label', shape=num_classes))))

# Function for coordinating training
def train(reader_train, epoch_size, max_epochs):
    set_computation_network_trace_level(0)
    input_var = input_variable((num_channels, image_height, image_width))
    label_var = input_variable((num_classes))

    z = create_model(input_var, 3, num_classes) # 3 for 20-layer, 8 for 50-layer
    lr_per_mb = [0.001]+[0.01]*80+[0.001]*40+[0.0001]

    # loss and metric
    ce = cross_entropy_with_softmax(z, label_var)
    pe = classification_error(z, label_var)

    minibatch_size = 16
    momentum_time_constant = -minibatch_size/np.log(0.9)
    l2_reg_weight = 0.0001
    
    lr_per_sample = [lr/minibatch_size for lr in lr_per_mb]
    lr_schedule = learning_rate_schedule(lr_per_sample, epoch_size=epoch_size, unit=UnitType.sample)
    mm_schedule = momentum_as_time_constant_schedule(momentum_time_constant)
    
    learner = momentum_sgd(z.parameters, lr_schedule, mm_schedule,
                           l2_regularization_weight = l2_reg_weight,
                           unit_gain=True)
    trainer = Trainer(z, ce, pe, learner)

    input_map = {input_var: reader_train.streams.features,
                 label_var: reader_train.streams.labels}

    log_number_of_parameters(z) ; print()
    progress_printer = ProgressPrinter(tag='Training')

    for epoch in range(max_epochs):
        sample_count = 0
        while sample_count < epoch_size:
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size-sample_count), input_map=input_map)
            trainer.train_minibatch(data)
            sample_count += trainer.previous_minibatch_sample_count
            progress_printer.update_with_trainer(trainer, with_metric=True)
        progress_printer.epoch_summary(with_metric=True)
        trainer.save_checkpoint(os.path.join(model_path, 'resnet20_{}.dnn'.format(epoch)))
        #z.save_model(os.path.join(model_path, 'resnet20_{}.dnn'.format(epoch)))
    return

if __name__ == '__main__':
    reader_train = create_reader(os.path.join(data_path, 'map.txt'))
    train(reader_train, epoch_size=10000, max_epochs=300)