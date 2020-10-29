import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from scipy.ndimage.filters import convolve
import cv2

import dnnlib
from dnnlib import EasyDict
import dnnlib.tflib as tflib

from networks_progan import *

def load_model(model_path):
    tflib.init_tf()
    with open(model_path, 'rb') as f:
        _G, _D, Gs = pickle.load(f)
    return _G, _D, Gs

def get_intermediate_outputs(latents, Gs_model):
    activations = {}
    rgbs = {}
    n = latents.shape[0]
    
    # 4x4 block: dense layer
    x = pixel_norm(latents)
    w = Gs_model.get_var('4x4/Dense/weight')
    b = Gs_model.get_var('4x4/Dense/bias')[0]
    x = tf.matmul(x, w) + b
    x = tf.reshape(x, [n, 512, 4, 4])
    x = leaky_relu(pixel_norm(x))
    activations['4x4/Dense'] = x
    
    # 4x4 block: conv2d layer
    w = Gs_model.get_var('4x4/Conv/weight')
    b = Gs_model.get_var('4x4/Conv/bias')[0]
    x1 = pixel_norm(leaky_relu(tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW') + b))
    activations['4x4/Conv'] = x
    
    # 4x4 block: toRGB layer
    w = Gs_model.get_var('ToRGB_lod5/weight')
    b = Gs_model.get_var('ToRGB_lod5/bias')[0]
    rgb1 = tf.nn.conv2d(x1, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW') + b
    rgbs['lod5'] = rgb1
    
    # 8x8 block: conv2d_transpose layer
    w = Gs_model.get_var('8x8/Conv0_up/weight')
    w = tf.transpose(w, [0, 1, 3, 2]) # [kernel, kernel, fmaps_out, fmaps_in]
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    os = [n, 512, 8, 8]
    b = Gs_model.get_var('8x8/Conv0_up/bias')[0]
    x2 = pixel_norm(leaky_relu(tf.nn.conv2d_transpose(x1, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW') + b))
    activations['8x8/Conv0_up'] = x2
    
    # 8x8 block: conv2d layer
    w = Gs_model.get_var('8x8/Conv1/weight')
    b = Gs_model.get_var('8x8/Conv1/bias')[0]
    x3 = pixel_norm(leaky_relu(tf.nn.conv2d(x2, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW') + b))
    activations['8x8/Conv1'] = x3
    
    # 8x8 block: toRGB layer
    w = Gs_model.get_var('ToRGB_lod4/weight')
    b = Gs_model.get_var('ToRGB_lod4/bias')[0]
    rgb2 = tf.nn.conv2d(x3, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW') + b
    rgbs['lod4'] = rgb2
    
    # 16x16 block: conv2d_transpose layer
    w = Gs_model.get_var('16x16/Conv0_up/weight')
    w = tf.transpose(w, [0, 1, 3, 2]) # [kernel, kernel, fmaps_out, fmaps_in]
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    os = [n, 512, 16, 16]
    b = Gs_model.get_var('16x16/Conv0_up/bias')[0]
    x4 = pixel_norm(leaky_relu(tf.nn.conv2d_transpose(x3, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW') + b))
    activations['16x16/Conv0_up'] = x4
    
    # 16x16 block: conv2d layer
    w = Gs_model.get_var('16x16/Conv1/weight')
    b = Gs_model.get_var('16x16/Conv1/bias')[0]
    x5 = pixel_norm(leaky_relu(tf.nn.conv2d(x4, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW') + b))
    activations['16x16/Conv1'] = x5
    
    # 16x16 block: toRGB layer
    w = Gs_model.get_var('ToRGB_lod3/weight')
    b = Gs_model.get_var('ToRGB_lod3/bias')[0]
    rgb3 = tf.nn.conv2d(x5, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW') + b
    rgbs['lod3'] = rgb3
    
    # 32x32 block: conv2d_transpose layer
    w = Gs_model.get_var('32x32/Conv0_up/weight')
    w = tf.transpose(w, [0, 1, 3, 2]) # [kernel, kernel, fmaps_out, fmaps_in]
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    os = [n, 512, 32, 32]
    b = Gs_model.get_var('32x32/Conv0_up/bias')[0]
    x6 = pixel_norm(leaky_relu(tf.nn.conv2d_transpose(x5, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW') + b))
    activations['32x32/Conv0_up'] = x6
    
    # 32x32 block: conv2d layer
    w = Gs_model.get_var('32x32/Conv1/weight')
    b = Gs_model.get_var('32x32/Conv1/bias')[0]
    x7 = pixel_norm(leaky_relu(tf.nn.conv2d(x6, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW') + b))
    activations['32x32/Conv1'] = x7
    
    # 32x32 block: toRGB layer
    w = Gs_model.get_var('ToRGB_lod2/weight')
    b = Gs_model.get_var('ToRGB_lod2/bias')[0]
    rgb4 = tf.nn.conv2d(x7, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW') + b
    rgbs['lod2'] = rgb4
    
    # 64x64 block: conv2d_transpose layer
    w = Gs_model.get_var('64x64/Conv0_up/weight')
    w = tf.transpose(w, [0, 1, 3, 2]) # [kernel, kernel, fmaps_out, fmaps_in]
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    os = [n, 512, 64, 64]
    b = Gs_model.get_var('64x64/Conv0_up/bias')[0]
    x8 = pixel_norm(leaky_relu(tf.nn.conv2d_transpose(x7, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW') + b))
    activations['64x64/Conv0_up'] = x8
    
    # 64x64 block: conv2d layer
    w = Gs_model.get_var('64x64/Conv1/weight')
    b = Gs_model.get_var('64x64/Conv1/bias')[0]
    x9 = pixel_norm(leaky_relu(tf.nn.conv2d(x8, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW') + b))
    activations['64x64/Conv1'] = x9
    
    # 64x64 block: toRGB layer
    w = Gs_model.get_var('ToRGB_lod1/weight')
    b = Gs_model.get_var('ToRGB_lod1/bias')[0]
    rgb5 = tf.nn.conv2d(x9, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW') + b
    rgbs['lod1'] = rgb5
    
    # 128x128 block: conv2d_transpose layer
    w = Gs_model.get_var('128x128/Conv0_up/weight')
    w = tf.transpose(w, [0, 1, 3, 2]) # [kernel, kernel, fmaps_out, fmaps_in]
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    os = [n, 512, 128, 128]
    b = Gs_model.get_var('128x128/Conv0_up/bias')[0]
    x10 = pixel_norm(leaky_relu(tf.nn.conv2d_transpose(x9, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW') + b))
    activations['128x128/Conv0_up'] = x10
    
    # 128x128 block: conv2d layer
    w = Gs_model.get_var('128x128/Conv1/weight')
    b = Gs_model.get_var('128x128/Conv1/bias')[0]
    x11 = pixel_norm(leaky_relu(tf.nn.conv2d(x10, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW') + b))
    activations['128x128/Conv1'] = x11
    
    # 128x128 block: toRGB layer
    w = Gs_model.get_var('ToRGB_lod0/weight')
    b = Gs_model.get_var('ToRGB_lod0/bias')[0]
    rgb6 = tf.nn.conv2d(x11, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW') + b
    rgbs['lod0'] = rgb6
    
    return activations, rgbs  

def visualize_activation(activation, original, alpha=0.5):
    activation -= activation.min()
    activation *= (1.0 / activation.max())
    resized_activation = cv2.resize(activation, (original.shape[0], original.shape[1]), interpolation=cv2.INTER_CUBIC)
    resized_activation = np.expand_dims(resized_activation, axis=2)
    overlay = ((1 - alpha) * original + alpha * resized_activation * original).astype('int32')
    return overlay

def plot_images(images, figsize=(10, 10)):
    rows = int(np.ceil(np.sqrt(images.shape[0])))
    cols = images.shape[0] // rows
    figs, axs = plt.subplots(rows, cols, figsize=figsize)
    for i,image in enumerate(images):
        r, c = i % cols, i // cols
        axs[r,c].axis('off')
        axs[r,c].imshow(image)
    plt.show()

def visualize_activation(activation, original, alpha=0.5):
    activation -= activation.min()
    activation *= (1.0 / activation.max())
    resized_activation = cv2.resize(activation, 
                                    (original.shape[0], original.shape[1]), 
                                    interpolation=cv2.INTER_CUBIC)
    overlay = original.copy()
    overlay[:,:,1] = ((1 - alpha) * original[:,:,1] + \
                      alpha * (1 - resized_activation) * original[:,:,1]).astype('uint8')
    overlay[:,:,2] = ((1 - alpha) * original[:,:,1] + \
                      alpha * (1 - resized_activation) * original[:,:,1]).astype('uint8')
    return overlay
    
def get_obj_mask(image, thresh=75):
    image_mask = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')
    image_mask[image[:,:,0]<thresh] = 1
    image_mask[image[:,:,1]<thresh] = 1
    image_mask[image[:,:,2]<thresh] = 1
    return image_mask

def get_color_mask(image, color):
    hrange = np.zeros((2, 3))
    if color == 'dark_blue':
        hrange[0] = np.array([0, 30, 0])
        hrange[1] = np.array([20, 255, 255])
    elif color == 'teal':
        hrange[0] = np.array([20, 30, 0])
        hrange[1] = np.array([60, 255, 255])
    elif color == 'green':
        hrange[0] = np.array([60, 30, 0])
        hrange[1] = np.array([80, 255, 255])
    elif color == 'yellow':
        hrange[0] = np.array([80, 30, 0])
        hrange[1] = np.array([95, 255, 255])
    elif color == 'orange':
        hrange[0] = np.array([95, 30, 0])
        hrange[1] = np.array([110, 255, 255])
    elif color == 'red':
        hrange[0] = np.array([110, 30, 0])
        hrange[1] = np.array([150, 255, 255])
    elif color == 'purple':
        hrange[0] = np.array([150, 30, 0])
        hrange[1] = np.array([255, 255, 255])
    elif color == 'gray':
        hrange[0] = np.array([0, 0, 0])
        hrange[1] = np.array([255, 30, 255])
    else:
        print('Error: invalid color argument.')
        return
    
    obj_mask = get_obj_mask(image, thresh=95)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_hsv, hrange[0], hrange[1])
    mask = (mask / 255)
    
    mask = (mask * obj_mask).astype('uint8')
    return mask
    
def get_activation_mask(activation, thresh=0.5):
    activation = (activation - np.min(activation)) / np.max(activation)
    activation_mask = np.zeros(activation.shape, dtype='uint8')
    activation_mask[activation>thresh] = 1
    return activation_mask

def compute_iou(image_mask, activation_mask):
    activation_mask = cv2.resize(activation_mask, 
                                 (image_mask.shape[0], image_mask.shape[1]), 
                                 interpolation=cv2.INTER_NEAREST).astype('uint8')
    union = np.bitwise_or(image_mask, activation_mask)
    intersection = np.bitwise_and(image_mask, activation_mask)
    return len(np.where(intersection==1)[0]) / len(np.where(union==1)[0])

def compute_neuron_response(neuron, images, activations, image_mask_func, **kwargs):
    ious = []
    activations = activations.eval()
    for i,image in enumerate(images):
        activation = activations[i,neuron]
        obj_mask = image_mask_func(image, **kwargs)
        activation_mask = get_activation_mask(activation)
        iou = compute_iou(obj_mask, activation_mask)
        ious.append(iou)
    return ious
