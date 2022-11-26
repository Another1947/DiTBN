import tensorflow as tf
import tensorflow.contrib.layers as Ly
import numpy as np
import os
import cv2
from scipy.io import loadmat, savemat
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def FusionNet(im_ms, im_pan, num_bands=8, num_filters=32, reuse=False):
    weight_decay = 1e-5
    with tf.variable_scope('net'):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        ms1 = Ly.conv2d(im_ms, num_filters, kernel_size=1, stride=1, padding='SAME',
                        weights_initializer=Ly.variance_scaling_initializer(),
                        weights_regularizer=Ly.l2_regularizer(weight_decay),
                        activation_fn=tf.nn.relu)
        pan1 = Ly.conv2d(im_pan, num_filters, kernel_size=1, stride=1, padding='SAME',
                         weights_initializer=Ly.variance_scaling_initializer(),
                         weights_regularizer=Ly.l2_regularizer(weight_decay),
                         activation_fn=tf.nn.relu)
        ms2 = Ly.conv2d(ms1, num_filters, kernel_size=3, stride=1, padding='SAME',
                        weights_initializer=Ly.variance_scaling_initializer(),
                        weights_regularizer=Ly.l2_regularizer(weight_decay),
                        activation_fn=tf.nn.relu)
        pan2 = Ly.conv2d(pan1, num_filters, kernel_size=3, stride=1, padding='SAME',
                         weights_initializer=Ly.variance_scaling_initializer(),
                         weights_regularizer=Ly.l2_regularizer(weight_decay),
                         activation_fn=tf.nn.relu)
        ms3 = Ly.conv2d(ms2, num_filters, kernel_size=3, stride=1, padding='SAME',
                        weights_initializer=Ly.variance_scaling_initializer(),
                        weights_regularizer=Ly.l2_regularizer(weight_decay),
                        activation_fn=tf.nn.relu)
        pan3 = Ly.conv2d(pan2, num_filters, kernel_size=3, stride=1, padding='SAME',
                         weights_initializer=Ly.variance_scaling_initializer(),
                         weights_regularizer=Ly.l2_regularizer(weight_decay),
                         activation_fn=tf.nn.relu)
        stacked_fm1 = tf.concat([ms3, pan3], axis=3)
        stacked_fm1 = ResBlock(stacked_fm1, num_filters=2 * num_filters)
        stacked_fm2 = tf.concat([stacked_fm1, ms1, pan1], axis=3)
        stacked_fm2 = ResBlock(stacked_fm2, num_filters=4 * num_filters)
        stacked_fm3 = tf.concat([stacked_fm2, ms1, pan1], axis=3)
        stacked_fm3 = ResBlock(stacked_fm3, num_filters=6 * num_filters)
        down_stacked_fm3 = Ly.conv2d(stacked_fm3, num_outputs=num_filters, kernel_size=1, stride=1, padding='SAME',
                                     weights_initializer=Ly.variance_scaling_initializer(),
                                     weights_regularizer=Ly.l2_regularizer(weight_decay),
                                     activation_fn=tf.nn.relu)
        detail = Ly.conv2d(down_stacked_fm3, num_bands, kernel_size=3, stride=1, padding='SAME',
                           weights_initializer=Ly.variance_scaling_initializer(),
                           weights_regularizer=Ly.l2_regularizer(weight_decay),
                           activation_fn=None)
        # fused_img = tf.add(ms, detail)
        return detail


def ResBlock(stacked_fm, num_filters):
    weight_decay = 1e-5
    # for i in range(num_resblocks):
    stacked_fm0 = Ly.conv2d(stacked_fm, num_outputs=64, kernel_size=1, stride=1, padding='SAME',
                            weights_initializer=Ly.variance_scaling_initializer(),
                            weights_regularizer=Ly.l2_regularizer(weight_decay),
                            activation_fn=tf.nn.relu)
    stacked_fm1 = Ly.conv2d(stacked_fm0, num_outputs=64, kernel_size=3, stride=1, padding='SAME',
                            weights_initializer=Ly.variance_scaling_initializer(),
                            weights_regularizer=Ly.l2_regularizer(weight_decay),
                            activation_fn=tf.nn.relu)
    stacked_fm2 = Ly.conv2d(stacked_fm1, num_filters, kernel_size=3, stride=1, padding='SAME',
                            weights_initializer=Ly.variance_scaling_initializer(),
                            weights_regularizer=Ly.l2_regularizer(weight_decay),
                            activation_fn=None)
    fused_fm = tf.add(stacked_fm, stacked_fm2)
    return fused_fm


def get_edge(data):
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.GaussianBlur(data[i, :, :], (7, 7), 2)
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.GaussianBlur(data[i, :, :, :], (7, 7), 2)
    return rs


if __name__ == '__main__':
    '''Import Data'''
    test_data = './dataset/train_test/WV3_simulated_test_set_with_lrms.mat'
    model_path = './models/WV3_3conv_pedge_spect41_improve2_11/Iter-100000'
    tf.reset_default_graph()
    data = loadmat(test_data)

    up_ms = data['up_ms'][...]  # N, H, W, C6
    pan = data['pan'][...]  # N, H, W
    up_ms_norm = np.array(up_ms, dtype=np.float32) / 2047.
    pan_norm = np.array(pan, dtype=np.float32) / 2047.

    # up_ms_norm_edge = get_edge(up_ms_norm)
    pan_norm_edge = get_edge(pan_norm)
    pan_norm_edge = pan_norm_edge[:, :, :, np.newaxis]

    N, H, W, C = up_ms.shape
    up_ms_p = tf.placeholder(dtype=tf.float32, shape=[N, H, W, C])
    # up_ms_edge = tf.placeholder(dtype=tf.float32, shape=[N, H, W, C])
    pan_edge = tf.placeholder(dtype=tf.float32, shape=[N, H, W, 1])

    detail = FusionNet(up_ms_p, pan_edge)
    hms = up_ms_p + detail
    output = tf.clip_by_value(hms, 0, 1)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    print("Testing: the model2_11 of Iter-100000")

    '''Start Test'''
    with tf.Session() as sess:
        sess.run(init)

        if tf.train.get_checkpoint_state(model_path):
            ckpt = tf.train.latest_checkpoint(model_path)
            saver.restore(sess, ckpt)
            print("Loading the trained model...")
        else:  # if there exists no trained model, use pre-trained model
            ckpt = tf.train.get_checkpoint_state(model_path + "pre-trained/")
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Load pre-trained model...")
        start_time = datetime.datetime.now()  # 开始时间
        fused_img = sess.run(fetches=output, feed_dict={up_ms_p: up_ms_norm, pan_edge: pan_norm_edge})
        end_time = datetime.datetime.now()
        cost_time = end_time - start_time
        savemat('./result/simulated2/general_simulated_fused_iter2e4_model2_11_WV3_1.mat', mdict={'general_simulated_fused_iter2e4_model2_11_no_ls': fused_img})

        print("Training finished! Time cost for training model: ", str(cost_time))
        print('Finished!')

