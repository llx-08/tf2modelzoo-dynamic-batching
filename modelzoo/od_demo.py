import pathlib
import time
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


def load_model(model_name):
    print("model name: ", model_name)
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)

    model_dir = pathlib.Path(model_dir) / "saved_model"

    model = tf.saved_model.load(str(model_dir))

    return model


def load_img2batch(img, batch):
    batched_img = []
    for i in range(batch):
        pass


def run_inference(model, image, batch_size):
    # return output_dict, pre-process time, inference time
    t = time.time()

    image = np.asarray(image)
    img_list = []
    for i in range(batch_size):
        img_list.append(image)

    t_ = time.time()

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(img_list, dtype=tf.uint8)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    # input_tensor = input_tensor[tf.newaxis, ...]
    print("input tensor shape: ", input_tensor.shape)
    # Run inference
    model_fn = model.signatures['serving_default']

    inference_t = time.time()
    output_dict = model_fn(input_tensor)
    inference_t_ = time.time()

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict, t_-t, inference_t_-inference_t


def show_inference(model, image_path, batch):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict, preprocess_time, inference_time = run_inference(model, image_np, batch_size=batch)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    plt.imshow(image_np)
    plt.savefig("11_.png")
    plt.show()
    display(Image.fromarray(image_np))
    return preprocess_time, inference_time


# List of the strings that is used to add correct label for each box.
PATH_TO_COCO_LABELS = '../workspace/training_demo/annotations/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_COCO_LABELS, use_display_name=True)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('../workspace/training_demo/images/test/')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))

print("test img path: ", TEST_IMAGE_PATHS)

# detection test
prefix_path = "/workspace/tf_processing/workspace/training_demo/exported_models/"
model_list = [
    "exported_faster_rcnn_resnet50_v1_640x640_coco17_tpu-8",
    "exported_ssd_resnet101_v1_fpn_640x640_coco17_tpu_8",
    "exported_efficientdet_d1_coco17_tpu-32"
]

batch_list = [1, 2, 4, 8, 16]
for model_name in model_list:
    model = tf.saved_model.load(prefix_path+model_name+"/saved_model")
    model.summary()
    print("inputs: ", model.signatures['serving_default'].inputs)
    print("output_dtypes: ", model.signatures['serving_default'].output_dtypes)
    print("output_shapes: ", model.signatures['serving_default'].output_shapes)

    for batch in batch_list:
        p_t, i_t = show_inference(model, TEST_IMAGE_PATHS[0], batch)
