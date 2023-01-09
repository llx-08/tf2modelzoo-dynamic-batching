import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from PIL import Image

imgSize = 224


preprocessing_layer = tf.keras.Sequential([
    layers.Resizing(imgSize, imgSize),
    layers.Rescaling(1. / 255.),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

# 加载在ImageNet上训练后的resnet50网络模型
resnet50 = tf.keras.applications.ResNet50(weights='imagenet', include_top=True)

img = Image.open("../plane.png")

