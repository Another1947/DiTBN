import tensorflow as tf  # version:1.15.0
import tensorflow.contrib.layers as Ly
from scipy.io import loadmat, savemat
import numpy as np
import cv2
import os
import datetime
from MTF_filter import downgrade_images

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def FusionNet(im_ms, im_pan, num_bands=4, num_filters=32, reuse=False):
    weight_decay = 1e-5
    with tf.variable_scope('net'):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        ms1 = Ly.conv2d(im_ms, num_filters, kernel_size=1, stride=1, padding='SAME',
                        weights_initializer=Ly.variance_scaling_initializer(),
                        # keyword argument 'dtype'
                        weights_regularizer=Ly.l2_regularizer(weight_decay),
                        activation_fn=tf.nn.relu)  # output: (N, H, W, C)
        pan1 = Ly.conv2d(im_pan, num_filters, kernel_size=1, stride=1, padding='SAME',
                         weights_initializer=Ly.variance_scaling_initializer(),
                         weights_regularizer=Ly.l2_regularizer(weight_decay),
                         activation_fn=tf.nn.relu)
        ms2 = Ly.conv2d(ms1, num_filters, kernel_size=3, stride=1, padding='SAME',
                        weights_initializer=Ly.variance_scaling_initializer(),  # ?
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
        stacked_fm1 = ResBlock(stacked_fm1, num_filters=2*num_filters)
        stacked_fm2 = tf.concat([stacked_fm1, ms1, pan1], axis=3)
        stacked_fm2 = ResBlock(stacked_fm2, num_filters=4*num_filters)
        stacked_fm3 = tf.concat([stacked_fm2, ms1, pan1], axis=3)
        stacked_fm3 = ResBlock(stacked_fm3, num_filters=6*num_filters)
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


def vis_ms(ms):
    if ms.get_shape().as_list()[3] == 4:
        _, b, g, r = tf.split(ms, 4, axis=3)
        rgb_ms = tf.concat([r, g, b], axis=3)
    else:
        _, b, g, _, r, _, _, _ = tf.split(ms, 8, axis=3)
        rgb_ms = tf.concat([r, g, b], axis=3)
    return rgb_ms


def im_gradient(im):
    im_grad = np.zeros_like(im)
    # im_grad = tf.zeros_like(im)
    N = im.shape[0]
    for m in range(N):
        im_gradr, im_gradc = np.gradient(im[m, :, :, :], axis=(0, 1))
        im_grad[m, :, :, :] = np.sqrt(np.multiply(im_gradr, im_gradr) + np.multiply(im_gradc, im_gradc))
    return im_grad


def get_edge(data):
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.GaussianBlur(data[i, :, :], (7, 7), 2)
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.GaussianBlur(data[i, :, :, :], (7, 7), 2)
    return rs


def get_batch_data(data, batchsize):
    gt = data['gt'][...]
    up_ms = data['up_ms'][...]
    lrms = data['lrms'][...]
    pan = data['pan'][...]

    gt_norm = np.array(gt, dtype=np.float32) / 2047.
    up_ms_norm = np.array(up_ms, dtype=np.float32) / 2047.
    lrms_norm = np.array(lrms, dtype=np.float32) / 2047.
    pan_norm = np.array(pan, dtype=np.float32) / 2047.

    N = gt.shape[0]
    batch_index = np.random.randint(0, N, size=batchsize)
    gt_batch = gt_norm[batch_index, :, :, :]
    up_ms_batch = up_ms_norm[batch_index, :, :, :]
    lrms_batch = lrms_norm[batch_index, :, :, :]
    pan_batch = pan_norm[batch_index, :, :]
    # pan_batch = pan_batch[:, :, :, np.newaxis]
    # up_ms_batch_edge = get_edge(up_ms_batch)
    pan_batch_edge = get_edge(pan_batch)
    # up_ms_batch_edge, pan_batch_edge = get_egde1(up_ms_batch, pan_batch)
    pan_batch_edge = pan_batch_edge[:, :, :, np.newaxis]
    pan_batch = pan_batch[:, :, :, np.newaxis]
    return gt_batch, lrms_batch, up_ms_batch, pan_batch_edge, pan_batch  # up_ms_batch_edge,


# ms = tf.random_uniform([2, 32, 32, 4], minval=0, maxval=1, dtype=tf.float32)
# pan = tf.random_uniform([2, 32, 32, 1], minval=0, maxval=1, dtype=tf.float32)
# out = FusionNet(ms, pan, num_bands=4, num_filters=32, reuse=False)
if __name__ == '__main__':
    tf.reset_default_graph()
    epochs = 100
    train_batch_size = 16
    test_batch_size = 16
    patch_size = 32
    iterations = 100100
    grad_weight = 0.1
    spectral_weight = 1
    model_directory = './models2/Ikonos_1116'
    train_data_path = './dataset/train_test/train_set_with_lrms_s6.mat'
    valid_data_path = './dataset/train_test/valid_set_with_lrms_s6.mat'
    log_path = './log/Ikonos_1114/'
    restore = False
    method = 'Adam'

    '''Import Data'''
    train_data = loadmat(train_data_path)
    valid_data = loadmat(valid_data_path)

    '''Pre allocate data usage space'''
    ground_truth = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, patch_size, patch_size, 8])
    up_ms = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, patch_size, patch_size, 4])
    # up_ms_edge = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, patch_size, patch_size, 4])
    # lms = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, patch_size // 4, patch_size // 4, 4])
    pan_edge = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, patch_size, patch_size, 1])
    pan_grad = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, patch_size, patch_size, 1])
    hms_grad = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, patch_size, patch_size, 4])
    lrms = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, patch_size/4, patch_size/4, 4])
    down_hms = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, patch_size/4, patch_size/4, 4])


    valid_ground_truth = tf.placeholder(dtype=tf.float32,
                                        shape=[test_batch_size, patch_size, patch_size, 4])  # N, H, W, C
    valid_up_ms = tf.placeholder(dtype=tf.float32, shape=[test_batch_size, patch_size, patch_size, 4])
    # valid_up_ms_edge = tf.placeholder(dtype=tf.float32, shape=[test_batch_size, patch_size, patch_size, 4])
    # valid_lms = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, patch_size // 4, patch_size // 4, 4])
    valid_pan_edge = tf.placeholder(dtype=tf.float32, shape=[test_batch_size, patch_size, patch_size, 1])
    valid_pan_grad = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, patch_size, patch_size, 1])
    valid_hms_grad = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, patch_size, patch_size, 4])
    valid_lrms = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, patch_size / 4, patch_size / 4, 4])
    valid_down_hms = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, patch_size / 4, patch_size / 4, 4])

    hms = FusionNet(up_ms, pan_edge)
    hms = up_ms + hms
    valid_hms = FusionNet(valid_up_ms, valid_pan_edge, reuse=True)
    valid_hms = valid_up_ms + valid_hms

    '''Loss Function'''
    mse = tf.reduce_mean(tf.square(hms - ground_truth)) + grad_weight * tf.reduce_mean(
        tf.square(hms_grad - pan_grad)) + spectral_weight * tf.reduce_mean(tf.square(down_hms - lrms))

    val_mse = tf.reduce_mean(tf.square(valid_hms - valid_ground_truth)) + grad_weight * tf.reduce_mean(
        tf.square(valid_hms_grad - valid_pan_grad)) + spectral_weight * tf.reduce_mean(tf.square(valid_down_hms - valid_lrms))

    '''Loss Summary'''
    train_loss_sum = tf.summary.scalar("train_loss", mse)
    valid_loss_num = tf.summary.scalar("valid_loss", val_mse)
    up_ms_sum = tf.summary.image("up_ms", tf.clip_by_value(vis_ms(up_ms), 0, 1))  #
    hms_sum = tf.summary.image("hms", tf.clip_by_value(vis_ms(hms), 0, 1))
    gt_sum = tf.summary.image("gt", tf.clip_by_value(vis_ms(ground_truth), 0, 1))
    all_sum = tf.summary.merge([train_loss_sum, up_ms_sum, hms_sum, gt_sum])

    '''Otimizer Setting'''
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='net')
    if method == "Adam":
        global_steps = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(0.001, global_steps, decay_steps=20000, decay_rate=0.5, staircase=True)
        optim_method = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999).minimize(mse,
                                                                                                 global_step=global_steps,
                                                                                                 var_list=train_vars)  # 0.001
    else:
        global_steps = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(0.1, global_steps, decay_steps=50000, decay_rate=0.1)
        clip_value = 0.1 / lr
        optim = tf.train.MomentumOptimizer(lr, 0.9)
        gradient, var = zip(*optim.compute_gradients(mse, var_list=train_vars))
        gradient, _ = tf.clip_by_global_norm(gradient, clip_value)
        optim_method = optim.apply_gradients(zip(gradient, var), global_step=global_steps)

    #'''GPU setting'''
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = False
    # sess = tf.Session(config=config)


    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)

        if restore:
            print("--->Loading Pretrained Model")
            pre_model = tf.train.get_checkpoint_state(model_directory)
            saver.restore(sess, pre_model.model_checkpoint_path)

        start_time = datetime.datetime.now()
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        train_log_path = os.path.join(log_path,
                                      'Ikonos_1114_train.log')
        train_log = open(train_log_path, "w")
        valid_log_path = os.path.join(log_path,
                                      'Ikonos_1114_valid.log')
        valid_log = open(valid_log_path, "w")

        train_mse = []
        valid_mse = []
        for i in range(iterations):
            train_gt, train_lrms, train_up_ms, train_pan_edge, train_pan = get_batch_data(train_data, batchsize=train_batch_size)
            train_pan_grad = im_gradient(
                train_pan)
            _, train_hms = sess.run([optim_method, hms],
                                    feed_dict={ground_truth: train_gt, up_ms: train_up_ms,
                                               pan_edge: train_pan_edge,
                                               })
            train_hms_grad = im_gradient(train_hms)
            down_train_hms = downgrade_images(train_hms, 4, 'IKONOS')
            train_loss = sess.run(mse, feed_dict={hms: train_hms,  ground_truth: train_gt, hms_grad: train_hms_grad,
                                                  pan_grad: train_pan_grad, down_hms: down_train_hms, lrms: train_lrms})
            if i % 100 == 0:
                print("===>Train Phase: Iter[%d/%d, %.2f%%], lr: %.8f, Loss: %.10f" % (
                    i, iterations, (100 * i / iterations), sess.run(lr), train_loss))
                train_log.write('Train Phase: Iter({}/{}) Loss: {:.10f}\n'.format(i, iterations, train_loss))
                train_mse.append(train_loss)

            if i % 1000 == 0 and i != 0:
                valid_gt_batch, valid_lrms_batch, valid_up_ms_batch, valid_pan_batch_edge, valid_pan_batch = get_batch_data(
                    valid_data, batchsize=test_batch_size)
                val_pan_grad = im_gradient(valid_pan_batch)
                val_hms = sess.run(valid_hms,
                                   feed_dict={valid_ground_truth: valid_gt_batch,
                                              valid_up_ms: valid_up_ms_batch,
                                              valid_pan_edge: valid_pan_batch_edge,
                                              })
                val_hms_grad = im_gradient(val_hms)
                down_valid_hms = downgrade_images(val_hms, 4, 'IKONOS')
                valid_loss = sess.run(val_mse, feed_dict={valid_hms: val_hms, valid_ground_truth: valid_gt_batch,
                                                          valid_hms_grad: val_hms_grad,
                                                          valid_pan_grad: val_pan_grad, valid_down_hms: down_valid_hms, valid_lrms: valid_lrms_batch})
                print("===>Valid Phase: Iter[%d/%d] Loss: %.10f" % (i, iterations, valid_loss))
                valid_log.write('Valid Phase: Iter({}/{}) Loss: {:.10f}\n'.format(i, iterations, valid_loss))
                valid_mse.append(valid_loss)

            if i % 20000 == 0 and i != 0:
                model_directory = model_directory
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)
                saver.save(sess, model_directory + '/Iter-' + str(i) + '/model-' + str(i) + '.ckpt')
                print("===>Model has been saved.")

        train_log.close()
        valid_log.close()
        savemat('./log/Ikonos_1114_no_ls.mat',
                mdict={'train_mse': train_mse, 'valid_mse': valid_mse})
        end_time = datetime.datetime.now()
        cost_time = end_time - start_time
        print("Training finished! Time cost for training model: ", str(cost_time))
