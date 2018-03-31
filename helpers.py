from __future__ import division
import math
import numpy as np
import tensorflow as tf
import scipy.misc
import itertools
import os
from glob import glob


# network related function and definitions

def _pop_batch_norm(x, pop_mean, pop_var, offset, scale):
    # private function to main batch_norm function
    return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, 1e-6)


def _batch_norm(x, pop_mean, pop_var, mean, var, offset, scale):
    # private function to main batch_norm function
    decay = 0.99
    
    dependency_1 = tf.assign(pop_mean, pop_mean * decay + mean * (1 - decay))
    dependency_2 = tf.assign(pop_var, pop_var * decay + var * (1 - decay))

    with tf.control_dependencies([dependency_1, dependency_2]):
        return tf.nn.batch_normalization(x, mean, var, offset, scale, 1e-6)


def batch_norm(x, is_training, name='batch_norm', is_linear= False):
    """Define function for batch normalization.

    Args:
        x: An input tensor
        is_training: parameter specified to use batch-norm in case of training
        name: user defined name used for variable scope.

    Returns: Output tensor after applying batch-norm.
    """ 
    with tf.variable_scope(name):
        depth = x.shape[-1]
        if is_linear:
            mean, var = tf.nn.moments(x, axes = [0])
        else:    
            mean, var = tf.nn.moments(x, axes = [0, 1, 2])
        
        var_init = tf.constant_initializer(0)
        offset = tf.get_variable("offset", [depth], tf.float32, var_init)
        var_init = tf.constant_initializer(1)
        scale = tf.get_variable("scale", [depth], tf.float32, var_init)
        
        pop_mean = tf.get_variable("pop_mean", [depth], initializer = tf.zeros_initializer(), trainable = False)
        pop_var = tf.get_variable("pop_var", [depth], initializer = tf.ones_initializer(), trainable = False)
        
        return tf.cond(
            is_training,
            lambda: _batch_norm(x, pop_mean, pop_var, mean, var, offset, scale),
            lambda: _pop_batch_norm(x, pop_mean, pop_var, offset, scale)
        )


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):

    """Define convolution function for 2D.

    Args:
        input_: An input tensor for convolution function.
        output_dim: Output dimension of the convolution kernal.
        k_h: height of the convolution kernal.
        k_w: width of the convolution kernal.
        d_h: height of the strides.
        d_w: height of the strides.
        stddev : user defined standard deviation for initialization.
        name: user defined name used for variable scope.

    Returns: Convolved tensor for the given input (image and kernal size)
    """ 
    with tf.variable_scope(name):

        # kernal : [height, width, output_channels, in_channels]
        weights = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(input_, weights, strides=[1, d_h, d_w, 1], padding='SAME') + biases

        return conv


def conv2d_transpose(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d_transpose", with_w=False):
    """Define deconvolution function for 2D.

    Args:
        input_: An input tensor for deconvolution function.
        output_dim: Output dimension of the deconvolution kernal.
        k_h: height of the convolution kernal.
        k_w: width of the convolution kernal.
        d_h: height of the strides.
        d_w: height of the strides.
        stddev : user defined standard deviation for initialization.
        name: user defined name used for variable scope.
        with_w: if the weight is also needed as output.

    Returns: Deconvolved tensor for the given input (image and kernal size).
    """
    with tf.variable_scope(name):

        # kernal : [height, width, output_channels, in_channels]
        weights = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))

        deconv = tf.nn.conv2d_transpose(input_, weights, output_shape=output_shape, strides=[1, d_h, d_w, 1]) + biases

        if with_w:
            return deconv, weights, biases
        else:
            return deconv


# def lrelu(x, name="lrelu"):
#     """Define leaky relu non-linearity function.

#     Args:
#         input_: An input tensor for leaky relu operation.
#         name: user defined name used for variable scope.

#     Returns: Output tensor (activation value) after applying lrelu non-linearity.
#     """
#     with tf.variable_scope(name):
#         return tf.nn.leaky_relu(x)

def lrelu(x, leak=0.2, name="lrelu"):
    """Define leaky relu non-linearity function.

    Args:
        input_: An input tensor for leaky relu operation.
        leak: value of offset or alpha required for leaky relu.
        name: user defined name used for variable scope.

    Returns: Output tensor (activation value) after applying lrelu non-linearity.
    """ 
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def relu(x, name="relu"):

    """Define relu non-linearity function using tf predefined relu.

    Args:
        input_: An input tensor for relu operation.
        name: user defined name used for variable scope.

    Returns: Output tensor after applying relu non-linearity.
    """
    with tf.variable_scope(name):
        return tf.nn.relu(x)


def linear(input_, output_size, scope=None, stddev=0.02, with_w=False):
    """Define lienar activation function used for fc layer.    
    
    Args:
        input_: An input tensor for activation function.
        output_dim: A output tensor size after passing through linearity.
        scope: variable scope, if None, used independently.
        stddev : user defined standard deviation for initialization.
        with_w: if the weight is also needed as output.

    Returns: logits of weights and biases.
    """
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(0.0))
        
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def flatten(x):
    """
    define a function to flatten a tensor

    Args:
        x: input tensor to be flattened

    Returns: flatten tensor
    """
    size = int(np.prod(x.shape[1:]))
    return tf.reshape(x, [-1, size])


# utility funtions and definitions
EXTENSIONS = ["png", "jpg", "jpeg"]

def image_files(path):
    """define a function to get the images from a directory as a list.

    Args:
        path: path for the input images.

    Returns: A list of all image files in the given directory".
    """
    return list(itertools.chain.from_iterable(
        glob(os.path.join(path, "*.{}".format(ext))) for ext in EXTENSIONS))


def get_image(image_path):
    """define a function to get the images from a directory.

    Args:
        image_path: path for the input images.
        image_size: list for size of an image.

    Returns: transformed all image present in a directory.
    """
    image = imread(image_path)
    image = np.array(image)/127.5 - 1.

    return image


def save_images(images, size, image_path):
    """define a function to save the images to a directory.

    Args:
        images: input images.
        size: size of an image.
        image_path: location to save the images.

    Returns: store the image.
    """
    images = (images+1.)/2.

    return imsave(images, size, image_path)


def imread(path):
    """define a function to read the images .

    Args:
        path: location of the images.

    Returns: use scipy read function to real the image in RGB mode.
    """
    return scipy.misc.imread(path, mode='RGB').astype(np.float)


def merge(images, size):
    """define a function to merge images.

    Args:
        images: input images.
        size: size of an image.

    Returns: merged image.
    """
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img


def imsave(images, size, path):
    """define a function to save images.

    Args:
        images: input images.
        size: size of an image.
        path: input image path.

    Return: save the images using scipy function.
    """
    img = merge(images, size)
    return scipy.misc.imsave(path, (255*img).astype(np.uint8))

