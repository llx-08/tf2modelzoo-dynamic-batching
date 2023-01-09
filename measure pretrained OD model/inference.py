#!/usr/bin/env python
# coding: utf-8
"""
Object Detection From TF2 Saved Model
=====================================
"""

# %%
# This demo will take you through the steps of running an "out-of-the-box" TensorFlow 2 compatible
# detection model on a collection of images. More specifically, in this example we will be using
# the `Saved Model Format <https://www.tensorflow.org/guide/saved_model>`__ to load the model.

# %% Download the test images ~~~~~~~~~~~~~~~~~~~~~~~~ First we will download the images that we will use throughout
# this tutorial. The code snippet shown bellow will download the test images from the `TensorFlow Model Garden
# <https://github.com/tensorflow/models/tree/master/research/object_detection/test_images>`_ and save them inside the
# ``data/images`` folder.
import os
from preprocess import *
import tensorflow as tf
import warnings
import keras
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

IMAGE_PATHS = download_images()
LABEL_FILENAME = 'mscoco_label_map.pbtxt'
PATH_TO_LABELS = download_labels(LABEL_FILENAME)

# %%
# Load the model

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Label maps correspond index numbers to category names, so that when our convolution network
# predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility
# functions, but anything that returns a dictionary mapping integers to appropriate string labels
# would be fine.

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

warnings.filterwarnings('ignore')  # Suppress Matplotlib warnings


def do_inference_with_differ_model(model, batchsize):

    for image_path in IMAGE_PATHS:
        print('Running inference.py for {}... '.format(image_path), end='')

        t1 = time.time()
        input_tensor = load_image_into_tf_tensor(image_path, batchsize)
        t2 = time.time()
        print("load time:", t2-t1)
        # input_tensor = np.expand_dims(image_np, 0)
        detections = model(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}

        # print("result: ", detections)
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = np.array(Image.open(image_path)).copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

        plt.figure()
        plt.imshow(image_np_with_detections)
        plt.show()
        print('Done')


if __name__ == '__main__':
    MODEL_DATE = '20200711'
    MODEL_NAME = 'centernet_hg104_1024x1024_coco17_tpu-32'
    PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)
    PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
    # PATH_TO_SAVED_MODEL = "/workspace/tf_processing/centernet_hg104_512x512_coco17_tpu-8/saved_model/saved_model.pb"
    print('Loading model...')
    start_time = time.time()
    # Load saved model and build the detection function
    detect_fn = keras.models.load_model(PATH_TO_SAVED_MODEL)
    detect_fn.summary()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    batchsize = 16

    # do_inference_with_differ_model(detect_fn, 16)

    # print(type(detect_fn))
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="fit_logs/", histogram_freq=1)
    #
    # mnist = tf.keras.datasets.mnist
    #
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, x_test = x_train / 255.0, x_test / 255.0
    #
    # detect_fn.fit(x=x_train,
    #               y=y_train,
    #               epochs=5,
    #               validation_data=(x_test, y_test),
    #               callbacks=[tensorboard_callback])