# Copyright (c) Microsoft. All rights reserved.
#
# Modified by Mary Wahl from work by Patrick Buehler, cf.
# https://github.com/Microsoft/CNTK/blob/master/Examples/Image/Detection/FastRCNN/A2_RunCntk_py3.py
#
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
from cntk import Trainer, UnitType, load_model
from cntk.layers import Placeholder, Constant, Dense
from cntk.graph import find_by_name
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
from cntk.learner import momentum_sgd, learning_rate_schedule, momentum_as_time_constant_schedule
from cntk.ops import input_variable, parameter, cross_entropy_with_softmax, classification_error, combine
from cntk.ops.functions import CloneMethod
from cntk.utils import log_number_of_parameters, ProgressPrinter
import cntk.io.transforms as xforms
import numpy as np
import os, sys

output_model_folder = 'D:\\repo\\cntk\\'
map_file = 'D:\\balanced_training_set\\map.txt'

num_channels = 3
image_height = 224
image_width = 224
num_classes = 6
epoch_size = 44184  # 44184 is the total number of images in the training set
mb_size = 16
max_epochs = 50
model_file = "D:\\repo\\cntk\\AlexNet.model"

def create_reader(map_file):
    transforms = [xforms.crop(crop_type='randomside', side_ratio=0.85, jitter_type='uniratio'),
                  xforms.scale(width=image_width,
                               height=image_height,
                               channels=num_channels,
                               interpolations='linear'),
                  xforms.color(brightness_radius=0.2,
                               contrast_radius=0.2,
                               saturation_radius=0.2)]
    return(MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features = StreamDef(field='image', transforms=transforms, is_sparse=False),
        labels   = StreamDef(field='label', shape=num_classes, is_sparse=False)))))

def frcn_predictor(features, n_classes):
    loaded_model = load_model(model_file)
    feature_node = find_by_name(loaded_model, 'features')
    last_node    = find_by_name(loaded_model, 'h2_d')
    all_layers = combine([last_node.owner]).clone(CloneMethod.freeze, {feature_node: Placeholder()})

    feat_norm = features - Constant(114)
    fc_out = all_layers(feat_norm)
    z = Dense(num_classes)(fc_out)

    return(z)

def train_fast_rcnn(debug_output=False):
    # Create the minibatch source
    minibatch_source = create_reader(map_file)

    # Input variables denoting features, rois and label data
    image_input = input_variable((num_channels, image_height, image_width))
    label_input = input_variable((num_classes))

    # define mapping from reader streams to network inputs
    input_map = {
        image_input: minibatch_source.streams.features,
        label_input: minibatch_source.streams.labels}

    # Instantiate the Fast R-CNN prediction model and loss function
    frcn_output = frcn_predictor(image_input, num_classes)
    ce = cross_entropy_with_softmax(frcn_output, label_input)
    pe = classification_error(frcn_output, label_input)

    # Set learning parameters
    l2_reg_weight = 0.0005
    lr_per_sample = [0.00001] * 10 + [0.000001] * 5 + [0.0000001]
    momentum_time_constant = 10
    lr_schedule = learning_rate_schedule(lr_per_sample, unit=UnitType.sample)
    mm_schedule = momentum_as_time_constant_schedule(momentum_time_constant)

    # Instantiate the trainer object
    progress_writers = [ProgressPrinter(tag='Training', num_epochs=max_epochs)]
    learner = momentum_sgd(frcn_output.parameters,
                           lr_schedule,
                           mm_schedule,
                           l2_regularization_weight=l2_reg_weight)
    trainer = Trainer(frcn_output, (ce, pe), learner, progress_writers)

    # Get minibatches of images and perform model training
    print("Training Fast R-CNN model for %s epochs." % max_epochs)
    log_number_of_parameters(frcn_output)
    
    for epoch in range(max_epochs):
        sample_count = 0
        while sample_count < epoch_size:
            data = minibatch_source.next_minibatch(min(mb_size, epoch_size-sample_count),
                input_map=input_map)
            trainer.train_minibatch(data)
            sample_count += trainer.previous_minibatch_sample_count

        trainer.summarize_training_progress()
        frcn_output.save_model(os.path.join(output_model_folder,
                                            'withcrops_{}.dnn'.format(epoch+1)))

    return

if __name__ == '__main__':
    train_fast_rcnn()
