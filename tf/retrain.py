# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modified 2017 Microsoft Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a model using a given dataset."""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import functools

from tensorflow.python.ops import control_flow_ops
from deployment import model_deploy
from nets import resnet_v1  # Needed to be modified, see https://github.com/tensorflow/models/issues/533
from tensorflow.contrib.training.python.training import evaluation

slim = tf.contrib.slim

''' Enumerate the flags '''
tf.app.flags.DEFINE_string('train_dir',
                           'D:\\tf\\models',
                           'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_string('dataset_name', 'aerial', 'The name of the dataset to load.')
tf.app.flags.DEFINE_string('dataset_dir',
                           'D:\\combined\\train_subsample',
                           'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string('checkpoint_path',
                           'D:\\tf\\resnet_v1_50.ckpt',
                           'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', 'resnet_v1_50/logits',
                           'Comma-separated list of scopes of variables to exclude when restoring '
                           'from a checkpoint.')
tf.app.flags.DEFINE_string('trainable_scopes', 'resnet_v1_50/logits',
                           'Comma-separated list of scopes to filter the set of variables to train.'
                           'By default, None would train all the variables.')

tf.app.flags.DEFINE_integer('num_clones', 1, 'Number of model clones to deploy.')
tf.app.flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones.')
tf.app.flags.DEFINE_integer('num_readers', 4, 'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer('num_preprocessing_threads', 4, 'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer('log_every_n_steps', 10, 'The frequency with which logs are printed.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 600, 'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer('save_interval_secs', 600, 'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')
tf.app.flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')
tf.app.flags.DEFINE_float('label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.9, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0, 'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_integer('replicas_to_aggregate', 1, 'The number of gradients to collect before updating params.')
tf.app.flags.DEFINE_integer('batch_size', 32, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('max_number_of_steps', 4000, 'The maximum number of training steps.')

FLAGS = tf.app.flags.FLAGS

def get_image_and_class_count(dataset_dir, split_name):
    df = pd.read_csv(os.path.join(dataset_dir, 'dataset_split_info.csv'))
    image_count = len(df.loc[df['split_name'] == split_name].index)
    class_count = len(df['class_name'].unique())
    return(image_count, class_count)

def read_label_file(dataset_dir, filename='labels.txt'):
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'r') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index+1:]
    return(labels_to_class_names)

def mean_image_subtraction(image, means):
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return(tf.concat(axis=2, values=channels))

def get_preprocessing():
    def preprocessing_fn(image, output_height=224, output_width=224):
        ''' Resize the image and subtract "mean" RGB values '''
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        #image = tf.expand_dims(image, 0)

        temp_dim = np.random.randint(175, 223)
        distorted_image = tf.random_crop(image, [output_height, output_width, 3])
        distorted_image = tf.expand_dims(distorted_image, 0)
        resized_image = tf.image.resize_bilinear(distorted_image, [output_height, output_width], align_corners=False)
        resized_image = tf.squeeze(resized_image)
        resized_image.set_shape([output_height, output_width, 3])
        resized_image = tf.image.random_flip_left_right(resized_image)

        image = tf.to_float(resized_image)
        return(mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN]))
    return(preprocessing_fn)

def get_network_fn(num_classes, weight_decay=0.0):
    arg_scope = resnet_v1.resnet_arg_scope(weight_decay=weight_decay)
    func = resnet_v1.resnet_v1_50
    @functools.wraps(func)
    def network_fn(images):
        with slim.arg_scope(arg_scope):
            return func(images, num_classes)
    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size
    return(network_fn)

def _add_variables_summaries(learning_rate):
    summaries = []
    for variable in slim.get_model_variables():
        summaries.append(tf.summary.image(variable.op.name, variable))
    summaries.append(tf.summary.scalar(learning_rate, name='training/Learning Rate'))
    return(summaries)

def _get_init_fn():
    if (FLAGS.checkpoint_path is None) or (tf.train.latest_checkpoint(FLAGS.train_dir)):
        return None

    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
              if var.op.name.startswith(exclusion):
                    excluded = True
                    break
        if not excluded:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Fine-tuning from {}'.format(checkpoint_path))

    return(slim.assign_from_checkpoint_fn(checkpoint_path,
                                          variables_to_restore,
                                          ignore_missing_vars=False))

def _get_variables_to_train():
    scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]
    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return(variables_to_train)

def get_dataset(dataset_name, dataset_dir, image_count, class_count, split_name):
    slim = tf.contrib.slim
    items_to_descriptions = {'image': 'A color image.',
                             'label': 'An integer in range(0, class_count)'}
    file_pattern = os.path.join(dataset_dir, '{}_{}_*.tfrecord'.format(dataset_name, split_name))
    reader = tf.TFRecordReader
    keys_to_features = {'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
                        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
                        'image/class/label': tf.FixedLenFeature([], tf.int64,
                                                                default_value=tf.zeros([], dtype=tf.int64))}
    items_to_handlers = {'image': slim.tfexample_decoder.Image(),
                         'label': slim.tfexample_decoder.Tensor('image/class/label')}
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    labels_to_names = read_label_file(dataset_dir)
    return(slim.dataset.Dataset(data_sources=file_pattern,
                                reader=reader,
                                decoder=decoder,
                                num_samples=image_count,
                                items_to_descriptions=items_to_descriptions,
                                num_classes=class_count,
                                labels_to_names=labels_to_names,
                                shuffle=True))

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        deploy_config = model_deploy.DeploymentConfig(num_clones=FLAGS.num_clones,
                                                      clone_on_cpu=FLAGS.clone_on_cpu,
                                                      replica_id=0,
                                                      num_replicas=1,
                                                      num_ps_tasks=0)

        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        image_count, class_count = get_image_and_class_count(FLAGS.dataset_dir, 'train')
        dataset = get_dataset('aerial', FLAGS.dataset_dir, image_count, class_count, 'train')
        network_fn = get_network_fn(num_classes=(dataset.num_classes), weight_decay=FLAGS.weight_decay)
        image_preprocessing_fn = get_preprocessing()

        with tf.device(deploy_config.inputs_device()):
            provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                                      num_readers=FLAGS.num_readers,
                                                                      common_queue_capacity=20 * FLAGS.batch_size,
                                                                      common_queue_min=10 * FLAGS.batch_size)
            [image, label] = provider.get(['image', 'label'])
            image = image_preprocessing_fn(image, 224, 224)
            images, labels = tf.train.batch([image, label],
                                            batch_size=FLAGS.batch_size,
                                            num_threads=FLAGS.num_preprocessing_threads,
                                            capacity=5 * FLAGS.batch_size)
            labels = slim.one_hot_encoding(labels, dataset.num_classes)
            batch_queue = slim.prefetch_queue.prefetch_queue([images, labels], capacity=2 * deploy_config.num_clones)

        def clone_fn(batch_queue):
            images, labels = batch_queue.dequeue()
            logits, end_points = network_fn(images)
            logits = tf.squeeze(logits) # added -- does this help?
            slim.losses.softmax_cross_entropy(logits, labels, label_smoothing=FLAGS.label_smoothing, weights=1.0)
            return(end_points)

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        first_clone_scope = deploy_config.clone_scope(0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        end_points = clones[0].outputs
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity/' + end_point, tf.nn.zero_fraction(x)))
        for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        with tf.device(deploy_config.optimizer_device()):
            decay_steps = int(dataset.num_samples / FLAGS.batch_size * FLAGS.num_epochs_per_decay)
            learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                                       global_step,
                                                       decay_steps,
                                                       FLAGS.learning_rate_decay_factor,
                                                       staircase=True,
                                                       name='exponential_decay_learning_rate')
            optimizer = tf.train.RMSPropOptimizer(learning_rate,
                                                  decay=FLAGS.rmsprop_decay,
                                                  momentum=FLAGS.rmsprop_momentum,
                                                  epsilon=FLAGS.opt_epsilon)
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))



        variables_to_train = _get_variables_to_train()
        total_loss, clones_gradients = model_deploy.optimize_clones(clones, optimizer, var_list=variables_to_train)
        summaries.add(tf.summary.scalar('total_loss', total_loss))

        grad_updates = optimizer.apply_gradients(clones_gradients, global_step=global_step)
        update_ops.append(grad_updates)

        update_op = tf.group(*update_ops)
        train_tensor = control_flow_ops.with_dependencies([update_op], total_loss, name='train_op')

        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        slim.learning.train(train_tensor,
                            logdir=FLAGS.train_dir,
                            master='',
                            is_chief=True,
                            init_fn=_get_init_fn(),
                            summary_op=summary_op,
                            number_of_steps=FLAGS.max_number_of_steps,
                            log_every_n_steps=FLAGS.log_every_n_steps,
                            save_summaries_secs=FLAGS.save_summaries_secs,
                            save_interval_secs=FLAGS.save_interval_secs,
                            sync_optimizer=None)


if __name__ == '__main__':
    tf.app.run()