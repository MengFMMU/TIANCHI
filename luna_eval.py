#!/usr/bin/env python

import tensorflow as tf 
import numpy as np 

from tqdm import tqdm

import luna_input
import luna

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'eval',
                           """Directory where to write event logs """)
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_depth', 3,
                            """Image depth (z dimention), odd number is prefered.""")
tf.app.flags.DEFINE_integer('image_xy', 48,
                            """Image width and height (x, y dimention).""")
tf.app.flags.DEFINE_string('data_dir', '/Volumes/SPIDATA/TIANCHI/train_processed',
                           """Path to the luna training data directory.""")
tf.app.flags.DEFINE_string('csv_file', '/Volumes/SPIDATA/TIANCHI/csv/train/annotations.csv',
                           """Path nodule annotation csv file.""")
tf.app.flags.DEFINE_integer('min_nodule', 10,
                            """Minimum nodule diameter in mm.""")
tf.app.flags.DEFINE_integer('max_nodule', 100,
                            """Maximum nodule diameter in mm.""")
tf.app.flags.DEFINE_integer('grid_size', 1,
                            """Grid size in nodule candidates generation.""")
tf.app.flags.DEFINE_string('ckpt_file', 'train/model.ckpt-852',
                            """Path to checkpoint file.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_boolean('debug', False,
                            """Whether to show detailed information for debugging.""")
tf.app.flags.DEFINE_boolean('verbose', False,
                            """Whether to show some detailed information.""")


def eval():
    batch_size = FLAGS.batch_size
    with tf.Graph().as_default():
        # input
        depth = FLAGS.image_depth
        if depth % 2 == 0:
            depth += 1
        hz = int((depth - 1) / 2)
        width_height = FLAGS.image_xy
        eval_input = luna_input.LUNAEvalInput(FLAGS.data_dir, 
                                     FLAGS.csv_file, 
                                     min_nodule=FLAGS.min_nodule, 
                                     max_nodule=FLAGS.max_nodule,
                                     batch_size=FLAGS.batch_size,
                                     sample_size_xy=width_height,
                                     grid_size=FLAGS.grid_size,
                                     sample_size_hz=hz,
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
        # inference
        logits = luna.inference(images)
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # run graph in session
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            
            saver.restore(sess, FLAGS.ckpt_file)

            for i in range(len(eval_input.seriesuids)):
                eval_input.next_scan()
                origin = eval_input.scan['origin']
                spacing = eval_input.scan['spacing']
                seriesuid = eval_input.seriesuid
                print('processing %s' % seriesuid)
                score_volume = np.zeros_like(eval_input.scan['image'], dtype=np.float32)
                nb_batch = int(np.ceil(float(eval_input.nb_candidates / FLAGS.batch_size)))
                for j in tqdm(range(nb_batch)):
                    batch_coords, batch_images = eval_input.next_batch()
                    logits_value = sess.run(logits, 
                        feed_dict={
                            input_images: batch_images,
                        })
                    exp_logits = np.exp(logits_value)
                    probability = exp_logits.T / np.sum(exp_logits, axis=1)
                    nodule_probability = probability.T[:,1]
                    zs, ys, xs = batch_coords[:,0], batch_coords[:,1], batch_coords[:,2]
                    score_volume[zs, ys, xs] = nodule_probability
                np.savez('%s/%s.npz' % (FLAGS.eval_dir, seriesuid), 
                    score_volume=score_volume,
                    spacing=spacing,
                    origin=origin)


def main(argv=None):
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    eval()


if __name__ == '__main__':
    tf.app.run()
