#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Linjian Zhang
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import random
import os
import cv2
import numpy as np
import scipy.io as sio
import shutil

########################################
# configuration

lr_base = 1e-3
epoch_max = 20
epoch_lr_decay = 500
epoch_save = 2
max_to_keep = 5

batch_size = 4
width = 512
height = 384

dir0 = '20170623'  # change it every time when training
net_name = 'flownet/'
# dir_restore = ''

dir_models = 'model/' + net_name
dir_logs = 'log/' + net_name
dir_model = dir_models + dir0
dir_log_train = dir_logs + dir0 + '_train'
dir_log_test = dir_logs + dir0 + '_test'

if not os.path.exists(dir_models):
    os.mkdir(dir_models)
if not os.path.exists(dir_logs):
    os.mkdir(dir_logs)
if os.path.exists(dir_model):
    shutil.rmtree(dir_model)
if os.path.exists(dir_log_train):
    shutil.rmtree(dir_log_train)
if os.path.exists(dir_log_test):
    shutil.rmtree(dir_log_test)

os.mkdir(dir_model)
os.mkdir(dir_log_train)
os.mkdir(dir_log_test)

########################################
# load image mean

dir_mean = 'data/mean_16338.mat'
mean_load = sio.loadmat(dir_mean)
mean = mean_load['mean']


########################################
# data process

def removeFile(directory_list):
    if '.directory' in directory_list:
        directory_list.remove('.directory')
    return directory_list


def load_data_train():

    return 0


def load_data_val():

    return 0


class DataBatch(object):
    def __init__(self, list, index, label, bs=batch_size, shuffle=True, minus_mean=True):
        self.list = list
        self.start_index = index  # [0, 5, 10, ...]
        self.label = np.array(label)
        self.bs = bs
        self.index = 0
        self.number = len(self.start_index)  # ~16330/5
        self.shuffle = shuffle
        self.minus_mean = minus_mean
        if self.shuffle: random.shuffle(self.start_index)

    def next_batch(self):
        start = self.index
        self.index += self.bs
        if self.index > self.number:
            if self.shuffle: random.shuffle(self.start_index)
            self.index = 0
            start = self.index
            self.index += self.bs
        end = self.index


########################################
class Net_Model(object):
    def __init__(self, constraints=False):
        self.x1 = tf.placeholder(tf.float32, [None, height, width, 3], name='x1')  # image1
        self.x2 = tf.placeholder(tf.float32, [None, height, width, 3], name='x2')  # image2
        self.x3 = tf.placeholder(tf.float32, [None, height, width, 2], name='x2')  # label
        self.x4 = tf.placeholder(tf.float32, [], name='x4')  # lr
        with tf.variable_scope('conv'):
            concat1 = tf.concat(3, [self.x1, self.x2])
            conv1 = slim.conv2d(concat1, 64, [7, 7], 2, scope='conv1')
            conv2 = slim.conv2d(conv1, 128, [5, 5], 2, scope='conv2')
            conv3 = slim.conv2d(conv2, 256, [5, 5], 2, scope='conv3')
            conv3_1 = slim.conv2d(conv3, 256, [3, 3], 1, scope='conv3_1')
            conv4 = slim.conv2d(conv3_1, 512, [3, 3], 2, scope='conv4')
            conv4_1 = slim.conv2d(conv4, 512, [3, 3], 1, scope='conv4_1')
            conv5 = slim.conv2d(conv4_1, 512, [3, 3], 2, scope='conv5')
            conv5_1 = slim.conv2d(conv5, 512, [3, 3], 1, scope='conv5_1')
            conv6 = slim.conv2d(conv5_1, 1024, [3, 3], 2, scope='conv6')
            conv6_1 = slim.conv2d(conv5_1, 1024, [3, 3], 1, scope='conv6_1')
            predict6 = slim.conv2d(conv6_1, 2, [3, 3], 1, scope='pred6')

        with tf.variable_scope('deconv'):
            # 12 * 16 flow
            deconv5 = slim.conv2d_transpose(conv6, 512, [4, 4], 2, scope='deconv5')
            deconvflow6 = slim.conv2d_transpose(predict6, 2, [4, 4], 2, 'SAME', scope='deconvflow6')
            concat5 = tf.concat(3, [conv5_1, deconv5, deconvflow6], name='concat5')
            predict5 = slim.conv2d(concat5, 2, [3, 3], 1, 'SAME', scope='predict5')
            # 24 * 32 flow
            deconv4 = slim.conv2d_transpose(concat5, 256, [4, 4], 2, 'SAME', scope='deconv4')
            deconvflow5 = slim.conv2d_transpose(predict5, 2, [4, 4], 2, 'SAME', scope='deconvflow5')
            concat4 = tf.concat(3, [conv4_1, deconv4, deconvflow5], name='concat4')
            predict4 = slim.conv2d(concat4, 2, [3, 3], 1, 'SAME', scope='predict4')
            # 48 * 64 flow
            deconv3 = slim.conv2d_transpose(concat4, 128, [4, 4], 2, 'SAME', scope='deconv3')
            deconvflow4 = slim.conv2d_transpose(predict4, 2, [4, 4], 2, 'SAME', scope='deconvflow4')
            concat3 = tf.concat(3, [conv3_1, deconv3, deconvflow4], name='concat3')
            predict3 = slim.conv2d(concat3, 2, [3, 3], 1, 'SAME', scope='predict3')
            # 96 * 128 flow
            deconv2 = slim.conv2d_transpose(concat3, 64, [4, 4], 2, 'SAME', scope='deconv2')
            deconvflow3 = slim.conv2d_transpose(predict3, 2, [4, 4], 2, 'SAME', scope='deconvflow3')
            concat2 = tf.concat(3, [conv2, deconv2, deconvflow3], name='concat2')
            predict2 = slim.conv2d(concat2, 2, [3, 3], 1, 'SAME', scope='predict2')

        with tf.variable_scope('loss'):
            flow6 = tf.image.resize_images(self.x3, [6, 8])
            loss6 = 0.32 * tf.reduce_mean(tf.abs(flow6 - predict6))
            flow5 = tf.image.resize_images(self.x3, [12, 16])
            loss5 = 0.08 * tf.reduce_mean(tf.abs(flow5 - predict5))
            flow4 = tf.image.resize_images(self.x3, [24, 32])
            loss4 = 0.02 * tf.reduce_mean(tf.abs(flow4 - predict4))
            flow3 = tf.image.resize_images(self.x3, [48, 64])
            loss3 = 0.01 * tf.reduce_mean(tf.abs(flow3 - predict3))
            flow2 = tf.image.resize_images(self.x3, [96, 128])
            loss2 = 0.005 * tf.reduce_mean(tf.abs(flow2 - predict2))
            self.loss = tf.add_n([loss6, loss5, loss4, loss3, loss2])
            tf.summary.scalar('loss6', loss6)
            tf.summary.scalar('loss5', loss5)
            tf.summary.scalar('loss4', loss4)
            tf.summary.scalar('loss3', loss3)
            tf.summary.scalar('loss2', loss2)
            tf.summary.scalar('total_loss', self.loss)
            self.merged = tf.merge_all_summaries()

        optimizer = tf.train.AdamOptimizer(self.x4)
        self.train_op = slim.learning.create_train_op(self.loss, optimizer)

        # init & save configuration
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)
        self.tvars = tf.trainable_variables()  # turn on if you want to check the variables
        self.variables_names = [v.name for v in self.tvars]
        self.init = tf.initialize_all_variables()

        # gpu configuration
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        # tf_config.gpu_options.visible_device_list = '1'


########################################
# main function
def main(_):
    # data preparation
    list, index, label = load_data_train()
    dataset_train = DataBatch(list, index, label, shuffle=True, minus_mean=True)
    list_v, index_v, label_v = load_data_val()
    dataset_val = DataBatch(list_v, index_v, label_v, shuffle=True, minus_mean=True)
    x1_v, x2_v, x1_1_v, x2_1_v, x3_v = dataset_val.next_batch()

    model = Net_Model()
    with tf.Session(config=model.tf_config) as sess:
        sess.run(model.init)
        writer_train = tf.train.SummaryWriter(dir_log_train, sess.graph)
        writer_val = tf.train.SummaryWriter(dir_log_test, sess.graph)

        for epoch in xrange(epoch_max):
            iter_max = len(index) // batch_size
            lr_decay = 0.1 ** (epoch / epoch_lr_decay)
            lr = lr_base * lr_decay
            for iter in xrange(iter_max):
                global_iter = epoch * iter_max + iter
                x1_batch, x2_batch, x3_batch = dataset_train.next_batch()

                feed_dict = {}
                feed_dict[model.x1] = x1_batch
                feed_dict[model.x2] = x2_batch
                feed_dict[model.x3] = x3_batch
                feed_dict[model.x4] = lr
                sess.run(model.train_op, feed_dict)

                # display
                if not (iter + 1) % 1:
                    merged_out_t, loss_out_t = sess.run([model.merged, model.loss], feed_dict)
                    writer_train.add_summary(merged_out_t, global_iter + 1)
                    print('epoch %03d, iter %04d, lr %.6f, loss: %.5f' % (epoch + 1, iter + 1, lr, loss_out_t))

                if not (iter + 1) % 5:
                    feed_dict_v = {}
                    feed_dict_v[model.x1] = x1_v
                    feed_dict_v[model.x2] = x2_v
                    feed_dict_v[model.x3] = x3_v
                    merged_out_v, loss_out_v = sess.run([model.merged, model.loss], feed_dict_v)
                    writer_val.add_summary(merged_out_v, global_iter + 1)
                    print('epoch %03d, iter %04d, ****val loss****: %.5f' % (epoch + 1, iter + 1, loss_out_v))

                # save
                if not (global_iter + 1) % (epoch_save * iter_max):
                    model.saver.save(sess, (dir_model + '/model'), global_step=global_iter + 1)


if __name__ == "__main__":
    tf.app.run()
