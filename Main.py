# Transfer Learning Gram-matrix reconstruction
import tensorflow as tf
import numpy as np
import os
import urllib

class DeepMCR(object):
	def __init__(self, layers=4):
		# 1). check the existence of the weights file
		self.layers = layers
		if not os.path.isdir('./vgg_weights'):
			os.path.mkdir('./vgg_weights')
		if not os.path.isfile('./vgg_weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
			urllib.urlretrieve('https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 
								'./vgg_weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

