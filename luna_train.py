#!/usr/bin/env python


import tensorflow as tf 
import numpy as np
from math import pow

import luna
from luna_input import LUNATrainInput

import time
from datetime import datetime
from glob import glob

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('macro_batch_size', 10,
                            """Number of scan to process in a macro batch.""")
tf.app.flags.DEFINE_string('sample_ratio', '0.5,0.3,0.2,0.0,0.0',
                            """Sample ratio of nodule, tissue and border samples""")
tf.app.flags.DEFINE_string('train_dir', 'train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_float('learning_rate', 0.01,
                            """Initial learning rate.""")
tf.app.flags.DEFINE_integer('decay_steps', 10,
                            """Decay step for learning rate.""")
tf.app.flags.DEFINE_float('decay_factor', 0.98,
                            """Decay factor for learning rate.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 5,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_boolean('shuffle', True,
                            """Whether to shuffle the batch.""")
tf.app.flags.DEFINE_boolean('load_ckpt', False,
                            """Whether to load checkpoint file.""")
tf.app.flags.DEFINE_integer('ckpt_step', 0,
                            """Global step of ckpt file.""")
tf.app.flags.DEFINE_boolean('save_false_samples', True,
                            """Whether to save false samples.""")
tf.app.flags.DEFINE_boolean('feed_false_samples', False,
                            """Whether to feed false samples.""")
tf.app.flags.DEFINE_string('false_sample_dir', 'false_samples',
                            """Path to save false samples.""")
tf.app.flags.DEFINE_string('false_sample_date', None,
                            """Date of false samples.""")
tf.app.flags.DEFINE_integer('false_sample_size', 10000,
                            """Number of false samples per file.""")



def train():
    batch_size = FLAGS.batch_size
    sample_ratio = map(float, FLAGS.sample_ratio.split(','))
    # may save false samples
    false_sample_images = None
    false_sample_labels = None
    false_sample_count = 0
    if FLAGS.save_false_samples:
        tf.gfile.MakeDirs(FLAGS.false_sample_dir)

    with tf.Graph().as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # input
        depth = FLAGS.image_depth
        if depth % 2 == 0:
            depth += 1
        hz = int((depth - 1) / 2)
        width_height = FLAGS.image_xy
        train_input = LUNATrainInput(FLAGS.data_dir, 
                                     FLAGS.csv_file, 
                                     min_nodule=FLAGS.min_nodule, 
                                     max_nodule=FLAGS.max_nodule,
                                     micro_batch_size=FLAGS.batch_size,
                                     macro_batch_size=FLAGS.macro_batch_size,
                                     sample_ratio=sample_ratio,
                                     sample_size_xy=width_height,
                                     sample_size_hz=hz,
                                     false_sample_dir=FLAGS.false_sample_dir,
                                     false_sample_date=FLAGS.false_sample_date,
                                     feed_false_samples=FLAGS.feed_false_samples,
                                     debug=FLAGS.debug,
                                     verbose=FLAGS.verbose)
        if FLAGS.use_fp16:
            FP = tf.float16
        else:
            FP = tf.float32
        input_images = tf.placeholder(FP, 
                shape=[batch_size, depth, width_height, width_height], name='input_image')
        # Convert from [batch, depth, height, width] to [batch, height, width, depth].
        images = tf.transpose(input_images, [0, 2, 3, 1], name='image')
        labels = tf.placeholder(tf.int64, shape=[batch_size], name='label')
        # Display the training images in the visualizer.
        _bs, _h, _w, _d = images.get_shape()
        
        images_slices = tf.slice(images, [0,0,0,int(_d/2)], [int(_bs),int(_h),int(_w),1],
            name='show_images_1')
        if depth >= 3:
            rgb_images = tf.slice(images, [0,0,0,0], [int(_bs),int(_h),int(_w),3],
                name='show_images_0')
            tf.summary.image('rgb_images', rgb_images, max_outputs=10)
        tf.summary.image('image_slices', images_slices, max_outputs=10)

        # inference
        logits = luna.inference(images)

        # calculate accuracy and error rate
        correct_prediction = tf.equal(tf.argmax(logits,1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        error_rate = 1 - accuracy
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('error_rate', error_rate)

        # calculate FP/FN ratio
        _positive = tf.equal(labels, 1)
        _negative = tf.equal(labels, 0)
        TP = tf.reduce_sum(tf.multiply(tf.cast(correct_prediction, tf.float32), 
                         tf.cast(_positive, tf.float32)))
        TN = tf.reduce_sum(tf.multiply(tf.cast(correct_prediction, tf.float32),
                         tf.cast(_negative, tf.float32)))
        FN_ratio = 1. - tf.divide(TP, tf.reduce_sum(tf.cast(_positive, tf.float32)))
        FP_ratio = 1. - tf.divide(TN, tf.reduce_sum(tf.cast(_negative, tf.float32)))
        tf.summary.scalar('FP_ratio', FP_ratio)
        tf.summary.scalar('FN_ratio', FN_ratio)

        # train to minimize loss
        loss = luna.loss(logits, labels)
        lr = tf.placeholder(tf.float64, name='leaning_rate')
        tf.summary.scalar('learning_rate', lr)
        train_op = luna.train(loss, lr, global_step)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # run graph in session
        with tf.Session() as sess:
            init = tf.global_variables_initializer() # create an operation initializes all the variables
            sess.run(init)
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter('%s' % FLAGS.train_dir, sess.graph)
            
            if FLAGS.load_ckpt:
                ckpt_file = '%s/model.ckpt-%d' % \
                    (FLAGS.train_dir, FLAGS.ckpt_step)
                print('restore sess with %s' % ckpt_file)
                saver.restore(sess, ckpt_file)

            start = time.time()
            for step in range(FLAGS.max_steps):
                batch_images, batch_labels = train_input.next_micro_batch()
                if FLAGS.shuffle:
                    idx = np.arange(FLAGS.batch_size)
                    np.random.shuffle(idx)
                    batch_images = batch_images[idx]
                    batch_labels = batch_labels[idx]
                lr_value = FLAGS.learning_rate * pow(FLAGS.decay_factor, 
                    (step / FLAGS.decay_steps))
                _, err, g_step, loss_value, correctness, summary = sess.run(
                    [train_op, error_rate, global_step, loss, correct_prediction, merged], 
                    feed_dict={
                        labels: batch_labels,
                        input_images: batch_images,
                        lr: lr_value,
                    })
                if (~correctness).sum() > 0:  # false sample number > 0
                    if false_sample_count  == 0:
                        false_sample_images = batch_images[~correctness]
                        false_sample_labels = batch_labels[~correctness]
                    else:
                        false_sample_images = np.concatenate((
                            false_sample_images, batch_images[~correctness]), axis=0)
                        false_sample_labels = np.concatenate((
                            false_sample_labels, batch_labels[~correctness]), axis=0)
                    false_sample_count += (~correctness).sum()
                    if false_sample_count >= FLAGS.false_sample_size:
                        time_str = time.strftime('%m-%d-%H-%M-%S')
                        np.savez('%s/false_samples-%s.npz' % (FLAGS.false_sample_dir, time_str), 
                            false_sample_images=false_sample_images,
                            false_sample_labels=false_sample_labels)
                        false_sample_count = 0

                if step % FLAGS.log_frequency == 0 and step != 0:
                    end = time.time()
                    duration = end - start
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)
                    format_str = ('%s: step %d, loss = %.2f, err = %.4f (%.1f examples/sec; %.3f '
                        'sec/batch)')
                    print (format_str % (datetime.now(), g_step, loss_value, err, 
                               examples_per_sec, sec_per_batch))
                    writer.add_summary(summary, g_step)
                    # Save the variables to disk.
                    saver.save(sess, '%s/model.ckpt' % FLAGS.train_dir,  global_step=g_step)
                    start = end


def main(argv=None):  # pylint: disable=unused-argument
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()