__author__ = 'Pawel'
# Copyright 2014 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################
# This script makes batches suitable for training from raw ILSVRC 2012 tar files.

import tarfile
from PIL import Image
from StringIO import StringIO
from random import shuffle
import sys
from time import time
import struct
import ctypes as ct
import transforms3d.euler as euler
from pyext._MakeDataPyExt import resizeJPEG
import itertools
import os
import cPickle
import scipy.io
import numpy as np
import math
import argparse as argp

# Set this to True to crop images to square. In this case each image will be
# resized such that its shortest edge is OUTPUT_IMAGE_SIZE pixels, and then the
# center OUTPUT_IMAGE_SIZE x OUTPUT_IMAGE_SIZE patch will be extracted.
#
# Set this to False to preserve image borders. In this case each image will be
# resized such that its shortest edge is OUTPUT_IMAGE_SIZE pixels. This was
# demonstrated to be superior by Andrew Howard in his very nice paper:
# http://arxiv.org/abs/1312.5402
CROP_TO_SQUARE = True
OUTPUT_IMAGE_SIZE = 768

# Number of threads to use for JPEG decompression and image resizing.
NUM_WORKER_THREADS = 8

# Don't worry about these.
OUTPUT_BATCH_SIZE = 1024
OUTPUT_SUB_BATCH_SIZE = 1024


def pickle(filename, data):
    with open(filename, "w") as fo:
        cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)


def unpickle(filename):
    fo = open(filename, 'r')
    contents = cPickle.load(fo)
    fo.close()
    return contents


def partition_list(l, partition_size):
    divup = lambda a, b: (a + b - 1) / b
    return [l[i * partition_size:(i + 1) * partition_size] for i in xrange(divup(len(l), partition_size))]


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_depth_image(filename):
    f = open(filename, 'r')
    width = struct.unpack('i', f.read(4))[0]
    height = struct.unpack('i', f.read(4))[0]
    max_size = width * height
    p = 0
    result = []
    for x in range(height):
        result.append([])
        for y in range(width):
            result[x].append(0)
    while p < max_size:
        num_empty = struct.unpack('i', f.read(4))[0]
        p += num_empty

        num_full = struct.unpack('i', f.read(4))[0]
        for x in range(num_full):
            result[int(p / width)][int(p % width)] = struct.unpack('h', f.read(2))[0]
            p += 1
    f.close()
    return np.asarray(result, dtype=np.int8)


def write_batches(target_dir, name, start_batch_num, label_filenames, image_filenames):
    image_filenames = partition_list(image_filenames, OUTPUT_BATCH_SIZE)
    label_filenames = partition_list(label_filenames, OUTPUT_BATCH_SIZE)
    makedir(target_dir)
    print "Writing %s batches..." % name
    size = 32, 32
    for i, (labels_batch, jpeg_file_batch) in enumerate(zip(label_filenames, image_filenames)):
        t = time()
        images_strings = []
        labels_strings = []
        for label in labels_batch:
            labels_strings.append(read_label(label))
        for image_filename in jpeg_file_batch:
            im = Image.fromarray(read_depth_image(image_filename), mode='L')
            im.save(image_filename[0:-3] + 'png', format='png')
            half_the_width = im.size[0] / 2
            half_the_height = im.size[1] / 2
            height = im.size[1]
            im = im.crop(
                (
                    half_the_width - half_the_height,
                    0,
                    half_the_width + half_the_height,
                    height
                )
            )
            im.thumbnail(size, Image.LINEAR)
            im.save(image_filename[0:-4] + '_cropped.png', format='png')
            images_strings.append(list(im.getdata()))
        batch_path = os.path.join(target_dir, 'data_batch_%d' % (start_batch_num + i))
        pickle(batch_path, {'data': images_strings, 'labels': labels_strings})
        print "Wrote %s (%s batch %d of %d) (%.2f sec)" % (batch_path, name, i + 1, len(image_filenames), time() - t)
    return i + 1



def my_str(number, decimals):
    result = str(number)
    result = (decimals - len(result)) * '0' + result
    return result


# class LazyFile:
#     def __init__(self, filename):
#         self.filename = filename
#
#     def read(self):
#         f = open(self.filename)
#         result = f.read()
#         f.flush()
#         f.close()
#         return result

def read_label(filename):
    label_file = open(filename, 'r')
    labels = label_file.read()
    label_file.flush()
    label_file.close()
    labels = labels.split('\n')
    # print labels
    labels = labels[0:3]
    labels = map(lambda x: x.split(' '), labels)
    # print labels
    labels = map(lambda x: map(float, x[0:3]), labels)
    # print labels
    rot_matrix = np.array(labels)
    eu = euler.mat2euler(rot_matrix, 'sxyz')
    pitch_max = 6
    yaw_max = 3
    yaw = int((eu[0] + math.pi/4) / (math.pi / yaw_max))
    pitch = int((eu[1] + math.pi/2) / (math.pi / pitch_max))
    label = pitch
    label *= yaw_max
    label += yaw
    f_lab = open(filename[0:-8] + "label.txt", "w")
    f_lab.write(str(eu) + '\n' + str(label))
    f_lab.close()
    # print labels
    return [label]


def make_set(_set, _labels, _jpeg_files):
    for i in _set:
        for f in os.listdir(args.src_dir + '/' + my_str(i, 2)):
            if f.endswith("_rgb.png"):
                # print f
                _jpeg_files.append(args.src_dir + '/' + my_str(i, 2) + '/' + f[0:-7] + "depth.bin")
                _labels.append(args.src_dir + '/' + my_str(i, 2) + '/' + f[0:-7] + "pose.txt")

if __name__ == "__main__":
    parser = argp.ArgumentParser()
    parser.add_argument('--src-dir',
                        help='Directory containing BIWI head pose dataset',
                        required=True)
    parser.add_argument('--tgt-dir',
                        help='Directory to output BIWI batches suitable for cuda-convnet to train on.',
                        required=True)
    args = parser.parse_args()

    print "CROP_TO_SQUARE: %s" % CROP_TO_SQUARE
    print "OUTPUT_IMAGE_SIZE: %s" % OUTPUT_IMAGE_SIZE
    print "NUM_WORKER_THREADS: %s" % NUM_WORKER_THREADS

    assert OUTPUT_BATCH_SIZE % OUTPUT_SUB_BATCH_SIZE == 0

    train_jpeg_files = []
    train_labels = []

    test_jpeg_files = []
    test_labels = []

    test_set = [1, 12]
    train_set = range(1, 24)
    map(lambda x: train_set.remove(x), test_set)
    # train_set = [2, 13]
    make_set(train_set, train_labels, train_jpeg_files)
    make_set(test_set, test_labels, test_jpeg_files)

    # print "done"

    # Write training batches
    i = write_batches(args.tgt_dir, 'training', 0, train_labels, train_jpeg_files)
    test_batch_start = i
    write_batches(args.tgt_dir, 'testing', test_batch_start, test_labels, test_jpeg_files)

    meta = unpickle('input_meta')
    meta_file = os.path.join(args.tgt_dir, 'batches.meta')
    meta.update({'batch_size': OUTPUT_BATCH_SIZE,
                 'num_vis': OUTPUT_IMAGE_SIZE**2,
                 'label_names': map(str, range(18))})
    pickle(meta_file, meta)

    print "All done! BIWI batches are in %s" % args.tgt_dir
