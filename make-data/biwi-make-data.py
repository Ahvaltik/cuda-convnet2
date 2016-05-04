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
from random import sample
import sys
from time import time
import struct
import transforms3d.euler as euler
import os
import cPickle
import scipy.io
import numpy as np
import skimage
import skimage.filter
import skimage.io
import skimage.transform
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

medium_depth = (490 + 1298) / 2

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
        locations = []
        for label in labels_batch:
            locations.append([])
            labels_strings.append(read_label(label, locations[-1]))
        location_iter = 0
        for image_filename in jpeg_file_batch:
            # im = Image.fromarray(read_depth_image(image_filename), mode='L')
            # im.thumbnail(size, Image.LINEAR)
            # im.save(image_filename[0:-4] + '_cropped.png', format='png')
            # arr = np.asarray(list(im.getdata()))

            arr = read_depth_image(image_filename, size, locations[location_iter])
            location_iter += 1
            images_strings.append(arr)
        batch_path = os.path.join(target_dir, 'data_batch_%d' % (start_batch_num + i))
        pickle(batch_path, {'data': images_strings, 'labels': labels_strings})
        print "Wrote %s (%s batch %d of %d) (%.2f sec)" % (batch_path, name, i + 1, len(image_filenames), time() - t)
    return i + 1


def make_elipse_filter(radius, min_val, max_val, med_val):
    elipse_filter = np.ones((radius*2, radius*2)) * med_val
    np.require(elipse_filter, dtype=np.uint16)
    depth = max_val - min_val
    for x in xrange(elipse_filter.shape[0]):
        for y in xrange(elipse_filter.shape[1]):
            r = ((x-radius)**2 + (y-radius)**2)**.5
            if r < radius:
                # elipse_filter[x][y] = min_val
                # elipse_filter[x][y] = max_val - ((1 - (r/(radius**2)))**.5) * depth
                elipse_filter[x][y] = int(max_val - (1 - (r/(radius**2))**.5) * depth)
    # np.require(elipse_filter, dtype=np.uint16)
    # Image.fromarray(elipse_filter, mode='I;16B').save('elipse.tiff', format='tiff')
    skimage.io.imsave('elipse.tiff', elipse_filter)
    return elipse_filter


def read_depth_image(filename, size, location):
    f = open(filename, 'r')
    width = struct.unpack('i', f.read(4))[0]
    height = struct.unpack('i', f.read(4))[0]
    max_size = width * height
    p = 0
    result = []
    # first_pos = None
    for x in xrange(height):
        result.append([])
        for y in xrange(width):
            # result[x].append(65535)
            result[x].append(medium_depth)
    while p < max_size:
        num_empty = struct.unpack('i', f.read(4))[0]
        p += num_empty
        # first_pos = int(p / width), int(p % width)
        num_full = struct.unpack('i', f.read(4))[0]
        for x in xrange(num_full):
            result[int(p / width)][int(p % width)] = struct.unpack('h', f.read(2))[0]
            p += 1
    f.close()
    arr = np.asarray(result, dtype=np.uint16)

    # # Face closest value
    # face_max_local = arr.max()
    # if face_max[0] < face_max_local:
    #     face_max[0] = face_max_local

    crop_size = 128
    Image.fromarray(arr, mode='I;16B').save(filename[0:-3] + 'tiff', format='tiff')
    cropped_arr = arr[crop_size: height - crop_size, crop_size: width - crop_size]
    nose_pos = cropped_arr.argmin()
    nose_pos = crop_size + nose_pos / (width - crop_size * 2), crop_size + nose_pos % (width - crop_size * 2)
    # print str(x) + ' ' + str(y)
    # best_cropped = None
    # smallest_difference = None
    # for x in xrange(crop_size, height - crop_size):
    #     for y in xrange(crop_size, width - crop_size):
    #         cropped = arr[x - crop_size: x + crop_size, y - crop_size: y + crop_size]
    #         difference = np.sum(np.square(np.absolute(cropped - elipse_filter)))
    #         if smallest_difference is None or smallest_difference < difference:
    #             smallest_difference = difference
    #             best_cropped = cropped

    # arr = best_cropped
    arr = arr[nose_pos[0] - crop_size: nose_pos[0] + crop_size, nose_pos[1] - crop_size: nose_pos[1] + crop_size]
    # arr = arr[first_pos[0] - (crop_size / 2): first_pos[0] + (crop_size * 3 / 2), first_pos[1] - (crop_size / 2): first_pos[1] + (crop_size * 3 / 2)]
    # print arr.shape
    # print first_pos
    # im = Image.fromarray(arr, mode='I;16B')

    # im.thumbnail(size, Image.LINEAR)
    Image.fromarray(arr, mode='I;16B').save(filename[0:-4] + '_cropped.tiff', format='tiff')
    # im.save('temp.tiff', format='tiff')
    # skimage.io.imread('temp.tiff', as_grey=True)
    # arr = skimage.img_as_float(skimage.transform.resize(skimage.filter.denoise_bilateral(arr, sigma_range=0.1, sigma_spatial=15), size))
    # skimage.io.imsave(filename[0:-4] + '_cropped.tiff', skimage.img_as_uint(arr))

    # im.save(filename[0:-4] + '_cropped.tiff', format='tiff')
    # arr = np.asarray(list(im.getdata()))
    arr = (arr - medium_depth * np.ones_like(arr)) / float(medium_depth - 490)
    return arr.reshape((arr.shape[0] * arr.shape[1]))


def my_str(number, decimals):
    result = str(number)
    result = (decimals - len(result)) * '0' + result
    return result


def crop_number(number, _min, _max):
    return max(_min, min(_max, number))


def read_label(filename, location):
    label_file = open(filename, 'r')
    labels = label_file.read()
    label_file.flush()
    label_file.close()
    labels = labels.split('\n')
    _location = map(float, labels[4].split(' ')[0:3])
    location.extend(_location)
    labels = map(lambda x: map(float, x[0:3]), map(lambda x: x.split(' '), labels[0:3]))
    # print labels
    rot_matrix = np.array(labels)
    eu = euler.mat2euler(rot_matrix, 'sxyz')
    pitch_max = 3
    yaw_max = 3
    yaw = int((eu[0] + math.pi/4) / (math.pi / (2 * yaw_max)))
    pitch = int((eu[1] + math.pi/2) / (math.pi / pitch_max))
    yaw = crop_number(yaw, 0, yaw_max - 1)
    pitch = crop_number(pitch, 0, pitch_max - 1)
    # for i in range(3):
    #     if location[i] < locationmin[i]:
    #         locationmin[i] = location[i]
    #     if location[i] > locationmax[i]:
    #         locationmax[i] = location[i]
    label = pitch
    label *= yaw_max
    label += yaw
    f_lab = open(filename[0:-8] + "label.txt", "w")
    f_lab.write(str(eu) + '\n' + str(label))
    f_lab.close()
    # print labels
    return [label]


def make_set(_set, _labels, _jpeg_files):
    temp_tuples = []
    for i in _set:
        for f in os.listdir(args.src_dir + '/' + my_str(i, 2)):
            if f.endswith("_rgb.png"):
                # print f
                temp_tuples.append((args.src_dir + '/' + my_str(i, 2) + '/' + f[0:-7] + "depth.bin",
                                        args.src_dir + '/' + my_str(i, 2) + '/' + f[0:-7] + "pose.txt"))
    shuffled_lists = [list(tup) for tup in zip(*sample(temp_tuples, len(temp_tuples)))]
    _labels.extend(shuffled_lists[1])
    _jpeg_files.extend(shuffled_lists[0])


if __name__ == "__main__":
    # locationmin = [0, 0, 0]
    # locationmax = [0, 0, 0]
    # face_max = [490]

    elipse_filter = make_elipse_filter(128, 490, 1298, medium_depth)

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
                 'label_names': map(str, range(9))})
    pickle(meta_file, meta)

    # print face_max[0]

    print "All done! BIWI batches are in %s" % args.tgt_dir
