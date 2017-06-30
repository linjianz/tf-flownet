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
import struct
import scipy.io as sio


########################################
# configuration
start_number = 0
test_pairs_number = 64
use_gpu_1 = True
batch_size = 32
width = 512
height = 384
max_to_keep = 0

dir_restore = 'model/flownet_simple/20170629_1/model-3125'
dir_dataset = '/media/csc105/Data/dataset/FlyingChairs/data/'
dir_test = 'test/20170630_1/'

if not os.path.exists(dir_test):
    os.mkdir(dir_test)

########################################
# load image mean
# dir_mean = 'data/mean.mat'
# mean_load = sio.loadmat(dir_mean)
# mean = mean_load['mean']


########################################
# data process
def remove_file(directory_list):
    if '.directory' in directory_list:
        directory_list.remove('.directory')
    return directory_list


def load_data_test():
    img1_list_test = []
    img2_list_test = []
    flow_list_test = []
    namelist = remove_file(os.listdir(dir_dataset))
    namelist.sort()
    for i in range(start_number, start_number+test_pairs_number):
        flow_list_test.append(dir_dataset + namelist[3*i])
        img1_list_test.append(dir_dataset + namelist[3*i+1])
        img2_list_test.append(dir_dataset + namelist[3*i+2])

    assert len(img1_list_test) == len(img2_list_test)
    assert len(img1_list_test) == len(flow_list_test)
    return img1_list_test, img2_list_test, flow_list_test


class Data(object):
    def __init__(self, list1, list2, list3, bs=batch_size, shuffle=True, minus_mean=True):
        self.list1 = list1
        self.list2 = list2
        self.list3 = list3
        self.bs = bs
        self.index = 0
        self.number = len(self.list1)
        self.index_total = range(self.number)
        self.shuffle = shuffle
        self.minus_mean = minus_mean
        if self.shuffle: random.shuffle(self.index_total)

    def read_flow(self, name):
        f = open(name, "rb")
        data = f.read()
        f.close()
        width = struct.unpack('@i', data[4:8])[0]
        height = struct.unpack('@i', data[8:12])[0]
        flowdata = np.zeros((height, width, 2))
        for i in range(width*height):
            data_u = struct.unpack('@f', data[12+8*i:16+8*i])[0]
            data_v = struct.unpack('@f', data[16+8*i:20+8*i])[0]
            n = int(i / width)
            k = np.mod(i, width)
            flowdata[n, k, :] = [data_u, data_v]
        return flowdata

    def next_batch(self):
        start = self.index
        self.index += self.bs
        if self.index > self.number:
            if self.shuffle: random.shuffle(self.index_total)
            self.index = 0
            start = self.index
            self.index += self.bs
        end = self.index
        img1_batch = []
        img2_batch = []
        flow_batch = []
        for i in range(start, end):
            img1 = cv2.imread(self.list1[self.index_total[i]]).astype(np.float32)
            img1_batch.append(img1)
            img2 = cv2.imread(self.list2[self.index_total[i]]).astype(np.float32)
            img2_batch.append(img2)
            flow = self.read_flow(self.list3[self.index_total[i]])
            flow_batch.append(flow)

        return np.array(img1_batch), np.array(img2_batch), np.array(flow_batch)


########################################
class NetModel(object):
    def __init__(self, use_gpu_1=False):
        self.x1 = tf.placeholder(tf.float32, [None, height, width, 3], name='x1')  # image1
        self.x2 = tf.placeholder(tf.float32, [None, height, width, 3], name='x2')  # image2
        self.x3 = tf.placeholder(tf.float32, [None, height, width, 2], name='x3')  # label
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
            conv6_1 = slim.conv2d(conv6, 1024, [3, 3], 1, scope='conv6_1')
            predict6 = slim.conv2d(conv6_1, 2, [3, 3], 1, activation_fn=None, scope='pred6')

        with tf.variable_scope('deconv'):
            # 12 * 16 flow
            deconv5 = slim.conv2d_transpose(conv6_1, 512, [4, 4], 2, scope='deconv5')
            deconvflow6 = slim.conv2d_transpose(predict6, 2, [4, 4], 2, 'SAME', scope='deconvflow6')
            concat5 = tf.concat(3, [conv5_1, deconv5, deconvflow6], name='concat5')
            predict5 = slim.conv2d(concat5, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict5')
            # 24 * 32 flow
            deconv4 = slim.conv2d_transpose(concat5, 256, [4, 4], 2, 'SAME', scope='deconv4')
            deconvflow5 = slim.conv2d_transpose(predict5, 2, [4, 4], 2, 'SAME', scope='deconvflow5')
            concat4 = tf.concat(3, [conv4_1, deconv4, deconvflow5], name='concat4')
            predict4 = slim.conv2d(concat4, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict4')
            # 48 * 64 flow
            deconv3 = slim.conv2d_transpose(concat4, 128, [4, 4], 2, 'SAME', scope='deconv3')
            deconvflow4 = slim.conv2d_transpose(predict4, 2, [4, 4], 2, 'SAME', scope='deconvflow4')
            concat3 = tf.concat(3, [conv3_1, deconv3, deconvflow4], name='concat3')
            predict3 = slim.conv2d(concat3, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict3')
            # 96 * 128 flow
            deconv2 = slim.conv2d_transpose(concat3, 64, [4, 4], 2, 'SAME', scope='deconv2')
            deconvflow3 = slim.conv2d_transpose(predict3, 2, [4, 4], 2, 'SAME', scope='deconvflow3')
            concat2 = tf.concat(3, [conv2, deconv2, deconvflow3], name='concat2')
            predict2 = slim.conv2d(concat2, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict2')
            # 192 * 256 flow
            deconv1 = slim.conv2d_transpose(concat2, 64, [4, 4], 2, 'SAME', scope='deconv1')
            deconvflow2 = slim.conv2d_transpose(predict2, 2, [4, 4], 2, 'SAME', scope='deconvflow2')
            concat1 = tf.concat(3, [conv1, deconv1, deconvflow2], name='concat1')
            self.predict1 = slim.conv2d(concat1, 2, [3, 3], 1, 'SAME', activation_fn=None, scope='predict1')

        with tf.variable_scope('loss'):
            weight = [1.0/2, 1.0/4, 1.0/8, 1.0/16, 1.0/32, 1.0/32]
            flow6 = tf.image.resize_images(self.x3, [6, 8])
            loss6 = weight[5] * self.mean_loss(flow6, predict6)
            flow5 = tf.image.resize_images(self.x3, [12, 16])
            loss5 = weight[4] * self.mean_loss(flow5, predict5)
            flow4 = tf.image.resize_images(self.x3, [24, 32])
            loss4 = weight[3] * self.mean_loss(flow4, predict4)
            flow3 = tf.image.resize_images(self.x3, [48, 64])
            loss3 = weight[2] * self.mean_loss(flow3, predict3)
            flow2 = tf.image.resize_images(self.x3, [96, 128])
            loss2 = weight[1] * self.mean_loss(flow2, predict2)
            flow1 = tf.image.resize_images(self.x3, [192, 256])
            loss1 = weight[0] * self.mean_loss(flow1, self.predict1)
            self.loss = tf.add_n([loss6, loss5, loss4, loss3, loss2, loss1])
            tf.summary.scalar('loss6', loss6)
            tf.summary.scalar('loss5', loss5)
            tf.summary.scalar('loss4', loss4)
            tf.summary.scalar('loss3', loss3)
            tf.summary.scalar('loss2', loss2)
            tf.summary.scalar('loss1', loss1)
            tf.summary.scalar('loss', self.loss)
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
        if use_gpu_1:
            self.tf_config.gpu_options.visible_device_list = '1'

    def mean_loss(self, gt, predict):
        loss = tf.reduce_mean(tf.abs(gt-predict))
        return loss


########################################
# main function
def main(_):
    # data preparation
    list1_test, list2_test, list3_test = load_data_test()
    data_test = Data(list1_test, list2_test, list3_test, shuffle=False, minus_mean=False)

    model = NetModel(use_gpu_1=use_gpu_1)
    with tf.Session(config=model.tf_config) as sess:
        model.saver.restore(sess, dir_restore)
        for iteration in xrange(len(list1_test) // batch_size):
            x1_t, x2_t, x3_t = data_test.next_batch()
            feed_dict = dict()
            feed_dict[model.x1] = x1_t
            feed_dict[model.x2] = x2_t
            feed_dict[model.x3] = x3_t
            predict_out, loss_out_t = sess.run([model.predict1, model.loss], feed_dict)
            sio.savemat((dir_test + 'flow_batch_%d' % iteration), {'flow':predict_out})
            print('iter %04d, loss: %.5f' % (iteration + 1, loss_out_t))


if __name__ == "__main__":
    tf.app.run()
