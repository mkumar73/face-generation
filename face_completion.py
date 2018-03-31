import argparse
import os
import tensorflow as tf

from network import FACE_COMPLETION


# add arguments for the face completion network script

parser = argparse.ArgumentParser()

# number of iteration to run the face completion model
parser.add_argument('--num_iter', type=int, default=500)

# checkpoint directory for the stored model
parser.add_argument('--checkpointDir', type=str, default='checkpoint')

# output directory to store the completed images 
parser.add_argument('--out_dir', type=str, default='completed')

# interval to print the result on console 
parser.add_argument('--out_interval', type=int, default=50)

# mask type used to cover the image, only one implementation as of now i.e Center masking
parser.add_argument('--mask_type', type=str, default='center')

# total number of images in the directory
parser.add_argument('imgs', type=str, nargs='+')

args = parser.parse_args()

# check existence of checkpoint directory
assert(os.path.exists(args.checkpointDir))

# make output directory if doesn't exists
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)


# configure and start the tensorflow session for face completion
with tf.Session() as sess:
    fc = FACE_COMPLETION(sess, checkpoint_dir=args.checkpointDir)
    fc.complete(args)
