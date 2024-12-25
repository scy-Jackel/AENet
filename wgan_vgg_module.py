# -*- coding: utf-8 -*-



"""
################################################################################

1. discriminator
2. generator
3. pre trained VGG net

"""
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as tcl
import numpy as np

"""
module (discriminator, generator, pretrained vgg)
"""
def discriminator(image, name="discriminator", reuse = True):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        l1 = lrelu(conv2d(image, 64, s=1, name='d_conv_1'))
        l2 = lrelu(conv2d(l1, 64, s=2, name='d_conv_2'))
        l3 = lrelu(conv2d(l2, 128, s=1, name='d_conv_3'))
        l4 = lrelu(conv2d(l3, 128, s=2, name='d_conv_4'))
        l5 = lrelu(conv2d(l4, 256, s=1, name='d_conv_5'))
        l6 = lrelu(conv2d(l5, 256, s=2, name='d_conv_6'))

        fc1 = lrelu(fcn(l6, 1024, name='d_fc_1'))
        fc2 = fcn(fc1, 1,name='d_fc_2')
        
        return fc2
    
    
def generator(image, name="generator", reuse = True):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        l1 = tf.nn.relu(conv2d(image, 32, name='g_conv_1'))
        l2 = tf.nn.relu(conv2d(l1, 32, name='g_conv_2'))
        l3 = tf.nn.relu(conv2d(l2, 32, name='g_conv_3'))
        l4 = tf.nn.relu(conv2d(l3, 32, name='g_conv_4'))
        l5 = tf.nn.relu(conv2d(l4, 32, name='g_conv_5'))
        l6 = tf.nn.relu(conv2d(l5, 32, name='g_conv_6'))
        l7 = tf.nn.relu(conv2d(l6, 32, name='g_conv_7'))
        l8 = tf.nn.relu(conv2d(l7, 1, name='g_conv_8'))
        
        def vars():
            return [var for var in tf.global_variables() if 'g_' in var.name]
        
        return l8



def autoencoder(image, name = "AE", reuse=True):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        outputs1 = tf.layers.conv2d(image, 64, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1_1', use_bias=False) 
        outputs1 = tf.nn.relu(outputs1)
        outputs2 = tf.layers.conv2d(outputs1, 64, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1_2', use_bias=False) 
        outputs2 = tf.nn.relu(outputs2)

        outputs3 = tf.layers.max_pooling2d(outputs2, 2, 2, name='maxpool1')

        outputs4 = tf.layers.conv2d(outputs3, 128, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2_1', use_bias=False) 
        outputs4 = tf.nn.relu(outputs4)
        outputs5 = tf.layers.conv2d(outputs4, 128, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2_2', use_bias=False) 
        outputs5 = tf.nn.relu(outputs5)

        outputs6 = tf.layers.max_pooling2d(outputs5, 2, 2, name='maxpool2')

        outputs7 = tf.layers.conv2d(outputs6, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3_1', use_bias=False) 
        outputs7 = tf.nn.relu(outputs7)
        outputs8 = tf.layers.conv2d(outputs7, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3_2', use_bias=False) 
        outputs8 = tf.nn.relu(outputs8)

        outputs9 = tf.layers.conv2d(outputs8, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4_1', use_bias=False) 
        outputs9 = tf.nn.relu(outputs9)
        outputs10 = tf.layers.conv2d(outputs9, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4_2', use_bias=False) 
        outputs10 = tf.nn.relu(outputs10)

        outputs11 = tf.layers.conv2d(outputs10, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv5_1', use_bias=False) 
        outputs11 = tf.nn.relu(outputs11)
        outputs12 = tf.layers.conv2d(outputs11, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv5_2', use_bias=False) 
        outputs12 = tf.nn.relu(outputs12)

        outputs13 = tf.layers.conv2d(outputs12, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv6_1', use_bias=False) 
        outputs13 = tf.nn.relu(outputs13)
        outputs14 = tf.layers.conv2d(outputs13, 128, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv6_2', use_bias=False) 
        outputs14 = tf.nn.relu(outputs14)

        outputs15 = tf.layers.conv2d_transpose(outputs14, 128, 2,strides=(2,2), padding='valid', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv7', use_bias=False)
        
        outputs16 = tf.layers.conv2d(outputs15, 128, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv8_1', use_bias=False) 
        outputs16 = tf.nn.relu(outputs16)
        outputs17 = tf.layers.conv2d(outputs16, 64, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv8_2', use_bias=False) 
        outputs17 = tf.nn.relu(outputs17)

        outputs18 = tf.layers.conv2d_transpose(outputs17, 64, 2,strides=(2,2), padding='valid', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='deconv9', use_bias=False)
        
        outputs19 = tf.layers.conv2d(outputs18, 64, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv10_1', use_bias=False) 
        outputs19 = tf.nn.relu(outputs19)
        outputs20 = tf.layers.conv2d(outputs19, 32, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv10_2', use_bias=False) 
        outputs20 = tf.nn.relu(outputs20)

        outputs21 = tf.layers.conv2d(outputs20, 1, 1, padding='valid', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv11_1', use_bias=False) 

        # return output , inter.
        return outputs21, outputs10



class AE:
    def __init__(self, size=64, ae_path = './autoencoder_3072.npy'):
        self.size = size
        if os.path.exists(ae_path):
            print('ae path found.')
        else:
            print('ae path not found.')
        self.data_dict = np.load(ae_path, allow_pickle=True, encoding='latin1').item()

    def extract_feature(self, input):
        # input shape: [128, 64, 64, 1]  128 is batchsize
        print('DEBUG: IN extract_feature input shape:', input.shape)

        conv1_1 = self.conv_layer(input, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'maxpool1')
        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'maxpool2')
        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv4_1 = self.conv_layer(conv3_2, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        print('DEBUG OUTPUT SHAPE:', conv4_2.shape)

        return conv4_2


    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(conv)
            return relu

    def get_conv_filter(self, name):
        # AE/conv3_1/kernel
        key_np = 'AE/{}/kernel'.format(name)
        return tf.constant(self.data_dict[key_np], name="filter")


class Vgg19:
    def __init__(self, size = 64, vgg_path = '.'):
        self.size = size
        self.VGG_MEAN = [103.939, 116.779, 123.68]

        vgg19_npy_path = os.path.join(vgg_path, "vgg19.npy")
        if os.path.exists(vgg19_npy_path):
            print("vgg19 found.")
        else:
            print("vgg19 not found.")
        self.data_dict  = np.load(vgg19_npy_path,allow_pickle=True, encoding='latin1').item()
        print("npy file loaded")


    def extract_feature(self, rgb):
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [self.size, self.size, 1]
        assert green.get_shape().as_list()[1:] == [self.size, self.size, 1]
        assert blue.get_shape().as_list()[1:] == [self.size, self.size, 1]
        bgr = tf.concat(axis=3, values=[
            blue - self.VGG_MEAN[0],
            green - self.VGG_MEAN[1],
            red - self.VGG_MEAN[2],
        ])
        print(bgr.get_shape().as_list()[1:])
        assert bgr.get_shape().as_list()[1:] == [self.size, self.size, 3]


        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')
        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')
        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        conv3_4 = self.conv_layer(conv3_3, "conv3_4")
        pool3 = self.max_pool(conv3_4, 'pool3')
        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        conv4_4 = self.conv_layer(conv4_3, "conv4_4")
        pool4 = self.max_pool(conv4_4, 'pool4')
        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        conv5_4 = self.conv_layer(conv5_3, "conv5_4")
        return conv5_4


    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")




def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def conv2d(input_, output_dim, ks=3, s=1, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None)


def fcn(input_, n_weight, name = 'fcn'):
    with tf.variable_scope(name):
        flat_img = tcl.flatten(input_)
        fc = tcl.fully_connected(flat_img, n_weight, activation_fn=None)
        return fc