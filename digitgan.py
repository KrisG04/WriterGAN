"""
Created on Tue Mar  5 17:23:29 2019

@author: krish
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.examples.tutorials.mnist import input_data
print("Mnist data is being downloaded...")
mnist = input_data.read_data_sets("MNIST_data/")
mnist_images = mnist.train.images
mnist_labels = mnist.train.labels
print("Download complete!")

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev = 0.1))