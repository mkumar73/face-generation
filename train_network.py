#!/usr/bin/env python3
import argparse
import os
import scipy.misc
import tensorflow as tf

from network import FACE_COMPLETION

# add arguments for training the network

parser = argparse.ArgumentParser()

# Obligatory parameter to be specified while running training script
parser.add_argument('--dataset', type=str)

# specify the number of epochs to train the model
parser.add_argument('--epoch', type=int, default=20)

# specify learning rate for model, Discriminator and generator
parser.add_argument('--learning_rate', type=float, default=0.0003)

# batch size for training
parser.add_argument('--batch_size', type=int, default=1000)

# checkpoint directory to store the model parameters
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')

args = parser.parse_args()

# make checkpoint directory if doesn't exists
if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)


# configure and start the tensorflow session for training
with tf.Session() as sess:
    fc = FACE_COMPLETION(sess, batch_size=args.batch_size, checkpoint_dir=args.checkpoint_dir)
    fc.train(args)


