# Copyright (c) Microsoft. All rights reserved.
#
# Modified by Mary Wahl from work by Patrick Buehler, cf.
# https://github.com/Microsoft/CNTK/blob/master/Examples/Image/Detection/FastRCNN/A2_RunCntk_py3.py
#
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import cntk.io.transforms as xforms
from cntk.train.training_session import CheckpointConfig, training_session
import numpy as np
import os, sys, argparse, cntk
from PIL import Image

def create_reader(map_filename, image_height, image_width, num_channels,
                  num_classes):
    transforms = [xforms.crop(crop_type='randomside',
                              side_ratio=0.85,
                              jitter_type='uniratio'),
                  xforms.scale(width=image_width,
                               height=image_height,
                               channels=num_channels,
                               interpolations='linear'),
                  xforms.color(brightness_radius=0.2,
                               contrast_radius=0.2,
                               saturation_radius=0.2)]
    return(cntk.io.MinibatchSource(
        cntk.io.ImageDeserializer(map_filename, cntk.io.StreamDefs(
            features=cntk.io.StreamDef(
                field='image', transforms=transforms, is_sparse=False),
            labels=cntk.io.StreamDef(
                field='label', shape=num_classes, is_sparse=False)))))

def modify_model(pretrained_model_filename, features, num_classes):
    loaded_model = cntk.load_model(pretrained_model_filename)
    feature_node = cntk.logging.graph.find_by_name(loaded_model, 'features')
    last_node = cntk.logging.graph.find_by_name(loaded_model, 'h2_d')
    all_layers = cntk.ops.combine([last_node.owner]).clone(
        cntk.ops.functions.CloneMethod.freeze,
        {feature_node: cntk.ops.placeholder()})

    feat_norm = features - cntk.layers.Constant(114)
    fc_out = all_layers(feat_norm)
    z = cntk.layers.Dense(num_classes)(fc_out)

    return(z)

def main(map_filename, output_dir, pretrained_model_filename):
    ''' Retrain and save the existing AlexNet model '''
    num_epochs = 50
    mb_size = 16

    # Find the number of classes and the number of samples per epoch
    labels = set([])
    epoch_size = 0
    with open(map_filename, 'r') as f:
        for line in f:
            labels.add(line.strip().split('\t')[1])
            epoch_size += 1
        sample_image_filename = line.strip().split('\t')[0]
    num_classes = len(labels)
    num_minibatches = int(epoch_size // mb_size)

    # find the typical image dimensions
    image_height, image_width, num_channels = np.asarray(
        Image.open(sample_image_filename)).shape
    assert num_channels == 3, 'Expected to find images with 3 color channels'
    assert (image_height == 224) and (image_width == 224), \
        'Expected to find images of size 224 pixels x 224 pixels'

    # Create the minibatch source
    minibatch_source = create_reader(map_filename, image_height, image_width,
                                     num_channels, num_classes)

    # Input variables denoting features, rois and label data
    image_input = cntk.ops.input_variable(
        (num_channels, image_height, image_width))
    label_input = cntk.ops.input_variable((num_classes))

    # define mapping from reader streams to network inputs
    input_map = {image_input: minibatch_source.streams.features,
                 label_input: minibatch_source.streams.labels}

    # Instantiate the Fast R-CNN prediction model and loss function
    model = modify_model(pretrained_model_filename, image_input, num_classes)
    ce = cntk.losses.cross_entropy_with_softmax(model, label_input)
    pe = cntk.metrics.classification_error(model, label_input)

    # Set learning parameters
    l2_reg_weight = 0.0005
    lr_per_sample = [0.00001] * 10 + [0.000001] * 5 + [0.0000001]
    momentum_time_constant = 10
    lr_schedule = cntk.learners.learning_rate_schedule(lr_per_sample,
        unit=cntk.UnitType.sample)
    mm_schedule = cntk.learners.momentum_as_time_constant_schedule(
        momentum_time_constant)

    # Instantiate the trainer object
    progress_writers = [cntk.logging.progress_print.ProgressPrinter(
        tag='Training',
        num_epochs=num_epochs,
        freq=num_minibatches)]
    learner = cntk.learners.momentum_sgd(model.parameters,
        lr_schedule,
        mm_schedule,
        l2_regularization_weight=l2_reg_weight)
    trainer = cntk.Trainer(model, (ce, pe), learner, progress_writers)

    # Perform retraining and save the resulting model
    cntk.logging.progress_print.log_number_of_parameters(model)
    training_session(
        trainer=trainer,
        max_samples=num_epochs*epoch_size,
        mb_source=minibatch_source, 
        mb_size=mb_size,
        model_inputs_to_streams=input_map,
        checkpoint_config=CheckpointConfig(
            frequency=epoch_size,
            filename=os.path.join(output_dir,
                                  'retrained_checkpoint.model')),
        progress_frequency=epoch_size
    ).train()
    model.save(os.path.join(output_dir, 'retrained.model'))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
Retrains a pretrained Alexnet model to label aerial images according to land
use. 
''')
    parser.add_argument('-i', '--input_map_file', type=str, required=True,
                        help='MAP file listing training images and labels.')
    parser.add_argument('-o', '--output_dir',
                        type=str, required=True,
                        help='Output directory where model will be saved.')
    parser.add_argument('-p', '--pretrained_model_filename',
                        type=str, required=True,
                        help='Filepath of the pretrained AlexNet model.')
    args = parser.parse_args()

    # Ensure argument values are acceptable before proceeding
    assert os.path.exists(args.input_map_file), \
        'Input MAP file {} does not exist'.format(args.input_map_file)
    os.makedirs(args.output_dir, exist_ok=True)
    assert os.path.exists(args.pretrained_model_filename), \
        'Model file {} does not exist'.format(args.pretrained_model_filename)

    main(args.input_map_file, args.output_dir, args.pretrained_model_filename)