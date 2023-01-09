import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pprint import pprint
import numpy as np
import tensorflow as tf
gpu = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpu[1:], 'GPU')
tf.config.experimental.set_memory_growth(gpu[1], True)
import pathlib
import time
from PIL import Image
# from IPython.display import display

from object_detection.utils import ops as util_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as util_vis

# patch tf1 into "util.ops"
# patch the location of gfile
util_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile

# load label
category_index = label_map_util.create_category_index_from_labelmap(
    '../workspace/training_demo/annotations/mscoco_label_map.pbtxt', use_display_name=True)


def load_model(model_name):
    url = 'http://download.tensorflow.org/models/object_detection/tf2/20210210/'+model_name+'.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name, origin=url, untar=True)
    model_dir = pathlib.Path(model_dir)/"saved_model"
    return tf.saved_model.load(str(model_dir)).signatures['serving_default']


def load_img2tensor(image_path):
    image = np.array(Image.open(image_path))
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    return input_tensor


def load_img2batch(image_paths):
    batch = []
    for image in image_paths:
        batch.append(np.array(Image.open(image)))
    return tf.convert_to_tensor(batch, dtype='uint8')


def handle_output(out):
    """
    3. car
    1: person
    2: bicycle
    4: motorcycle
    6: bus
    8: trunk
    :param out:
    :return: num_detections of time items in above list
        e.g. [10, 2, 1, 0, 0, 2]
    """
    classes = out['detection_classes'].numpy().astype(int)
    classes = classes[:out['num_detections'].numpy().astype(int)[0]]
    return [np.sum(classes == i) for i in [3, 1, 2, 4, 6, 8]]


def gen_trace(model):
    model_fn = load_model(model)
    model_fn(load_img2tensor(r'../workspace/training_demo/images/train/acg.gy_01.jpg'))  # avoid cold start...

    # path = r'../dataset/M-30'
    # files = sorted(list(pathlib.Path(path).glob('*.jpg')))
    # t, shape, num = [], [], []
    # for img in files:
    #     _start = time.time()
    #     tensor = load_img2tensor(img)
    #     output = model_fn(tensor)
    #
    #     t.append(1000 * (time.time() - _start))
    #     shape.append(tensor.shape[1:3])
    #     num.append(handle_output(output))
    # with open(r'trace_m.csv', 'w') as f:
    #     for i in range(len(t)):
    #         f.write('{:.3f},{},{},{},{}\n'.format(t[i], shape[i][0], shape[i][1], num[i][0], num[i][1]))
    # print('finished')

    # ===============================
    path = r'../dataset/Urban1'
    files = sorted(list(pathlib.Path(path).glob('*.jpg')))
    t, num = [], []
    for img in files:
        _start = time.time()
        tensor = load_img2tensor(img)

        output = model_fn(tensor)

        t.append(1000 * (time.time() - _start))
        num.append(handle_output(output))
    with open(r'trace_u.csv', 'w') as f:
        for i in range(len(t)):
            f.write('{:.3f},{}\n'.format(t[i], num[i][0]))
    print('finished')


if __name__ == '__main__':

    # models from tf1 model zoo
    """
    Model name                  Speed(ms)  COCO mAP
    ssd_mobilenet_v1_fpn_coco   56          32
    ssd_resnet_50_fpn_coco      76          35
    faster_rcnn_resnet101_coco  106         32
    faster_rcnn_nas             1833        43
    """
    model_name = [
        'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
        'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
        'faster_rcnn_resnet101_coco_2018_01_28',
        'faster_rcnn_nas_coco_2018_01_28',
        # 'centernet_mobilenetv2fpn_512x512_coco17_od'  # tf2 model
    ]
    img = r'../workspace/training_demo/images/train/acg.gy_01.jpg'
    # gen_trace(model_name[0])

    load_img2batch([img])

    model_fn = load_model(model_name[0])
    model_fn(load_img2tensor(img))

    for i in [1, 2, 4, 8]:
        print('inference: batch size = {}'.format(i))
        batch = load_img2batch([img] * i)
        for j in range(10):
            _start = time.time()
            model_fn(batch)
            print('  infer-{} t={:.3f}ms'.format(j, 1000*(time.time()-_start)))



















