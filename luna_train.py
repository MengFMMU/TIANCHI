#!/usr/bin/env python


import tensorflow as tf 
import numpy as np

import luna
from luna_input import LUNATrainInput

import time
from datetime import datetime

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 10,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/Volumes/SPIDATA/TIANCHI/train_processed',
                           """Path to the luna training data directory.""")
tf.app.flags.DEFINE_string('csv_file', '/Volumes/SPIDATA/TIANCHI/csv/train/annotations.csv',
                           """Path nodule annotation csv file.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_string('train_dir', 'train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 5,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_integer('image_depth', 3,
                            """Image depth (z dimention), odd number is prefered.""")
tf.app.flags.DEFINE_integer('image_xy', 48,
                            """Image width and height (x, y dimention).""")


def train():
    batch_size = FLAGS.batch_size
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        # input
        depth = FLAGS.image_depth
        if depth % 2 == 0:
            depth += 1
        hz = int((depth - 1) / 2)
        width_height = FLAGS.image_xy
        train_input = LUNATrainInput(FLAGS.data_dir, FLAGS.csv_file, 30, 100,
                                     micro_batch_size=FLAGS.batch_size,
                                     sample_size_xy=width_height,
                                     sample_size_hz=hz)
        if FLAGS.use_fp16:
            FP = tf.float16
        else:
            FP = tf.float32
        input_images = tf.placeholder(FP, 
                shape=[batch_size, depth, width_height, width_height], name='input_image')
        # Convert from [batch, depth, height, width] to [batch, height, width, depth].
        images = tf.transpose(input_images, [0, 2, 3, 1], name='image')
        labels = tf.placeholder(FP, shape=[batch_size], name='label')
        # Display the training images in the visualizer.
        _bs, _h, _w, _d = images.get_shape()
        show_images = tf.slice(images, [0,0,0,int(_d/2)], [int(_bs),int(_h),int(_w),1],
            name='show_images')
        tf.summary.image('tf_images', images)
        tf.summary.image('show_images', show_images)

        # inference
        logits = luna.inference(images)
        # train to minimize loss
        loss = luna.loss(logits, labels)
        train_op = luna.train(loss)

        saver = tf.train.Saver()

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""
            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))
        # run graph in session
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
            config=tf.ConfigProto(
                log_device_placement=FLAGS.log_device_placement),
            save_checkpoint_secs=30,
            save_summaries_steps=30,) as mon_sess:
            
            # saver.restore(mon_sess, '%s/model.ckpt-0' % FLAGS.train_dir)
            while not mon_sess.should_stop():
                batch_images, batch_labels = train_input.next_micro_batch()
                _, labels_ = mon_sess.run([train_op, labels], feed_dict={
                    labels: batch_labels,
                    input_images: batch_images,
                    })
                print(labels_)

def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()