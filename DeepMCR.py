# Transfer Learning Gram-matrix reconstruction
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import urllib
import h5py
import Config
import scipy.io as sio
from scipy.optimize import fmin_l_bfgs_b
from pyDOE import lhs
from sklearn.cluster import KMeans
class DeepMCR(object):
	def __init__(self):
		# 1). check the existence of the weights file
		self.config = Config.Config()
		if not os.path.isdir('./vgg_weights'):
			os.mkdir('./vgg_weights')
		if not os.path.isfile('./vgg_weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'):
			print("[Model Pre-check] VGG weights not found. Downloading ...")
			urllib.urlretrieve('https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', 
								'./vgg_weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
			print("Done.")
		else:
			print("[Model Pre-check] VGG weights found.")
		self.loadWeights()
		print("[Progress] Weights loaded.")
		self.img = sio.loadmat(self.config.img)['IMG']
		if self.config.lhs:
			# use LHS to convert one phase to three phase
			lhd = lhs(3, samples=self.config.phaseNum, criterion='maximin')
			phaseMap = {}
			for i in xrange(self.config.phaseNum):
				phaseMap[str(i)] = lhd[i, :]

			temp = np.zeros((224, 224, 3))
			for i in xrange(224):
				for j in xrange(224):
					label = self.img[i, j]
					vectorRepresentation = phaseMap[str(label)]
					temp[i, j, :] = vectorRepresentation
			self.img = temp
			self.img[:, :, 0] -= 0.5
			self.img[:, :, 1] -= 0.5
			self.img[:, :, 2] -= 0.5

		else:
			# just copy and paste one channel into three channels
			temp = np.zeros((224, 224, 3))
			temp[:, :, 0] = self.img
			temp[:, :, 1] = self.img
			temp[:, :, 2] = self.img
			self.img = temp
		#substrate mean
			self.img = self.img.astype(np.float32)
			self.img[:, :, 0] -= 0.40760392
			self.img[:, :, 1] -= 0.45795686
			self.img[:, :, 2] -= 0.48501961
		self.img = np.expand_dims(self.img, 0)
		self.build()
		self.decode()

	def decode(self):
		if not self.config.lhs:
			#convert RGB to grayscale
			r, g, b = self.recon[:, :, 0], self.recon[:, :, 1], self.recon[:, :, 2]
			gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
			np.save(self.recon, gray)
			return gray

		else:
			#kmeans clustering based phase identification
			img = self.recon
			assert len(img.shape) == 3
			L1, L2, _ = img.shape
			X = np.reshape(img, (224*224, -1)) # transform into data sheet format
			kmeans = KMeans(n_clusters=phaseNum, max_iter=1000).fit(X)
			y = kmeans.labels_
			#centers = kmeans.cluster_centers_
			ret = np.reshape(y, (224, 224))
			np.save(self.recon, ret)
			return ret

	def loadWeights(self):
		# load VGG weights
		f = h5py.File('./vgg_weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5','r')
		ks = f.keys()
		self.vgg_weights=[]
		self.vgg_bias=[]
		for i in range(22):
			if (len(f[ks[i]].values())) != 0:     
				self.vgg_weights.append(f[ks[i]].values()[0][:])
				self.vgg_bias.append(f[ks[i]].values()[1][:])
			else:
				continue
		del f
		del ks
		# weights
		self.W_conv1 = tf.constant(self.vgg_weights[0])
		self.W_conv2 = tf.constant(self.vgg_weights[1])
		self.W_conv3 = tf.constant(self.vgg_weights[2])
		self.W_conv4 = tf.constant(self.vgg_weights[3])
		self.W_conv5 = tf.constant(self.vgg_weights[4])
		self.W_conv6 = tf.constant(self.vgg_weights[5])
		self.W_conv7 = tf.constant(self.vgg_weights[6])
		self.W_conv8 = tf.constant(self.vgg_weights[7])
		self.W_conv9 = tf.constant(self.vgg_weights[8])
		self.W_conv10 = tf.constant(self.vgg_weights[9])
		self.W_conv11 = tf.constant(self.vgg_weights[10])
		self.W_conv12 = tf.constant(self.vgg_weights[11])
		self.W_conv13 = tf.constant(self.vgg_weights[12])
		self.W_conv14 = tf.constant(self.vgg_weights[13])
		self.W_conv15 = tf.constant(self.vgg_weights[14])
		self.W_conv16 = tf.constant(self.vgg_weights[15])
		
		#biases
		self.b_conv1 = tf.reshape(tf.constant(self.vgg_bias[0]),[-1])
		self.b_conv2 = tf.reshape(tf.constant(self.vgg_bias[1]),[-1])
		self.b_conv3 = tf.reshape(tf.constant(self.vgg_bias[2]),[-1])
		self.b_conv4 = tf.reshape(tf.constant(self.vgg_bias[3]),[-1])
		self.b_conv5 = tf.reshape(tf.constant(self.vgg_bias[4]),[-1])
		self.b_conv6 = tf.reshape(tf.constant(self.vgg_bias[5]),[-1])
		self.b_conv7 = tf.reshape(tf.constant(self.vgg_bias[6]),[-1])
		self.b_conv8 = tf.reshape(tf.constant(self.vgg_bias[7]),[-1])
		self.b_conv9 = tf.reshape(tf.constant(self.vgg_bias[8]),[-1])
		self.b_conv10 = tf.reshape(tf.constant(self.vgg_bias[9]),[-1])
		self.b_conv11 = tf.reshape(tf.constant(self.vgg_bias[10]),[-1])
		self.b_conv12 = tf.reshape(tf.constant(self.vgg_bias[11]),[-1])
		self.b_conv13 = tf.reshape(tf.constant(self.vgg_bias[12]),[-1])
		self.b_conv14 = tf.reshape(tf.constant(self.vgg_bias[13]),[-1])
		self.b_conv15 = tf.reshape(tf.constant(self.vgg_bias[14]),[-1])
		self.b_conv16 = tf.reshape(tf.constant(self.vgg_bias[15]),[-1])
		del self.vgg_weights
		del self.vgg_bias

	def filterConfig(self, d):
		d1 = d.copy()
		for k in d1.keys():
			if self.config.layers[k] == 0:
				del d1[k]
		return d1


	def build(self):
		x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='x')
		xp = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='xp')
		E = self.LossGraph(x, xp)

		x_sample = np.random.randn(1, 224, 224, 3)
		x_init = np.reshape(x_sample, (224*224*3,))

		target = self.img
		assert x_sample.shape == target.shape
		g = tf.gradients(E, x, name='gradient_x')
		
		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()

		def getGradient(xx):
			xxt = xx.reshape((1,224,224,3))
			g_ret = sess.run([g], feed_dict={x:xxt, xp:target})
			ret = g_ret[0][0].reshape((224*224*3,))
			return ret.astype(np.float64).T
		def getFun(xx):
			xxt = xx.reshape((1,224,224,3))
			E_ret = sess.run([E], feed_dict={x:xxt, xp:target})
			return E_ret[0]
		
		gg = lambda xx: getGradient(xx)
		ff = lambda xx: getFun(xx)
		
		upperbound = np.min(target)
		lowerbound = np.max(target)
		bounds = [(upperbound, lowerbound) for i in range(224*224*3)]
		
		ret = fmin_l_bfgs_b(func=ff, x0=x_init, fprime=gg, bounds=bounds, maxiter=10000)
		recon = ret[0].reshape((224, 224, 3))
		#recon[:, :, 0] += 0.40760392
		#recon[:, :, 1] += 0.45795686
		#recon[:, :, 2] += 0.48501961

		#np.save(self.config.recon, recon)

		self.recon = recon

		
	def LossGraph(self, x, xp):

		x_activations = self.buildVGG(x)
		xp_activations = self.buildVGG(xp)

		x_activations = self.filterConfig(x_activations)
		xp_activations = self.filterConfig(xp_activations)


		def gram_matrix(v):
			# v is the activation on a layer
			assert isinstance(v, tf.Tensor)
			v.get_shape().assert_has_rank(4)
			dim = v.get_shape().as_list()
			v = tf.reshape(v, [dim[1] * dim[2], dim[3]])
			if dim[1] * dim[2] < dim[3]:
				return tf.matmul(v, v, transpose_b=True)
			else:
				return tf.matmul(v, v, transpose_a=True)
		E_total = 0.
		for k in x_activations.keys():
			G1 = gram_matrix(x_activations[k])
			G2 = gram_matrix(xp_activations[k])
			dims = x_activations[k].shape.as_list()
			E = tf.reduce_sum(tf.square(G1-G2), axis=[1,0]) /4. / (dims[3] ** 2) /  ((dims[1] * dims[2]) ** 2)
			E_total += E
		return E_total





	def buildVGG(self, x):

		def conv2d(x, W, stride, padding="SAME"):
			return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
		def max_pool(x, k_size, stride, padding="SAME"):
			# use avg pooling instead, as described in the paper
			return tf.nn.max_pool(x, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1], padding=padding) 

		# vgg block 1
		conv_out1 = conv2d(x, self.W_conv1, stride=1, padding='SAME')
		conv_out1 = tf.nn.bias_add(conv_out1, self.b_conv1)
		conv_out1 = tf.nn.relu(conv_out1)

		conv_out2 = conv2d(conv_out1, self.W_conv2, stride=1, padding='SAME')
		conv_out2 = tf.nn.bias_add(conv_out2, self.b_conv2)
		conv_out2 = tf.nn.relu(conv_out2)
		conv_out2_pool = max_pool(conv_out2, k_size=2, stride=2, padding="SAME")

		# vgg block 2
		conv_out3 = conv2d(conv_out2_pool, self.W_conv3, stride=1, padding='SAME')
		conv_out3 = tf.nn.bias_add(conv_out3, self.b_conv3)
		conv_out3 = tf.nn.relu(conv_out3)

		conv_out4 = conv2d(conv_out3, self.W_conv4, stride=1, padding='SAME')
		conv_out4 = tf.nn.bias_add(conv_out4, self.b_conv4)
		conv_out4 = tf.nn.relu(conv_out4)
		conv_out4_pool = max_pool(conv_out4, k_size=2, stride=2, padding="SAME")

		# vgg block 3

		conv_out5 = conv2d(conv_out4_pool, self.W_conv5, stride=1, padding='SAME')
		conv_out5 = tf.nn.bias_add(conv_out5, self.b_conv5)
		conv_out5 = tf.nn.relu(conv_out5)

		conv_out6 = conv2d(conv_out5, self.W_conv6, stride=1, padding='SAME')
		conv_out6 = tf.nn.bias_add(conv_out6, self.b_conv6)
		conv_out6 = tf.nn.relu(conv_out6)

		conv_out7 = conv2d(conv_out6, self.W_conv7, stride=1, padding='SAME')
		conv_out7 = tf.nn.bias_add(conv_out7, self.b_conv7)
		conv_out7 = tf.nn.relu(conv_out7)

		conv_out8 = conv2d(conv_out7, self.W_conv8, stride=1, padding='SAME')
		conv_out8 = tf.nn.bias_add(conv_out8, self.b_conv8)
		conv_out8 = tf.nn.relu(conv_out8)
		conv_out8_pool = max_pool(conv_out8, k_size=2, stride=2, padding='SAME')

		# vgg block 4
		conv_out9 = conv2d(conv_out8_pool, self.W_conv9, stride=1, padding='SAME')
		conv_out9 = tf.nn.bias_add(conv_out9, self.b_conv9)
		conv_out9 = tf.nn.relu(conv_out9)

		conv_out10 = conv2d(conv_out9, self.W_conv10, stride=1, padding='SAME')
		conv_out10 = tf.nn.bias_add(conv_out10, self.b_conv10)
		conv_out10 = tf.nn.relu(conv_out10)

		conv_out11 = conv2d(conv_out10, self.W_conv11, stride=1, padding='SAME')
		conv_out11 = tf.nn.bias_add(conv_out11, self.b_conv11)
		conv_out11 = tf.nn.relu(conv_out11)

		conv_out12 = conv2d(conv_out11, self.W_conv12, stride=1, padding='SAME')
		conv_out12 = tf.nn.bias_add(conv_out12, self.b_conv12)
		conv_out12 = tf.nn.relu(conv_out12)
		conv_out12_pool = max_pool(conv_out12, k_size=2, stride=2, padding='SAME')

		# vgg block 5
		conv_out13 = conv2d(conv_out12_pool, self.W_conv13, stride=1, padding='SAME')
		conv_out13 = tf.nn.bias_add(conv_out13, self.b_conv13)
		conv_out13 = tf.nn.relu(conv_out13)

		conv_out14 = conv2d(conv_out13, self.W_conv14, stride=1, padding='SAME')
		conv_out14 = tf.nn.bias_add(conv_out14, self.b_conv14)
		conv_out14 = tf.nn.relu(conv_out14)

		conv_out15 = conv2d(conv_out14, self.W_conv15, stride=1, padding='SAME')
		conv_out15 = tf.nn.bias_add(conv_out15, self.b_conv15)
		conv_out15 = tf.nn.relu(conv_out15)

		conv_out16 = conv2d(conv_out15, self.W_conv16, stride=1, padding='SAME')
		conv_out16 = tf.nn.bias_add(conv_out16, self.b_conv16)
		conv_out16 = tf.nn.relu(conv_out16)
		conv_out16_pool = max_pool(conv_out16, k_size=2, stride=2, padding='SAME')
		
		ret = {'conv_out1': conv_out1, 
			   'conv_out2': conv_out2, 
			   'conv_out2_pool': conv_out2_pool,
			   'conv_out3': conv_out3,
			   'conv_out4': conv_out4,
			   'conv_out4_pool': conv_out4_pool,
			   'conv_out5': conv_out5,
			   'conv_out6': conv_out6, 
			   'conv_out7': conv_out7,
			   'conv_out8': conv_out8,
			   'conv_out8_pool': conv_out8_pool,
			   'conv_out9': conv_out9, 
			   'conv_out10': conv_out10, 
			   'conv_out11': conv_out11, 
			   'conv_out12': conv_out12, 
			   'conv_out12_pool': conv_out12_pool,
			   'conv_out13': conv_out13,
			   'conv_out14': conv_out14, 
			   'conv_out15': conv_out15, 
			   'conv_out16': conv_out16, 
			   'conv_out16_pool': conv_out16_pool}
		return ret









	#def build(self):
		# build tf graph and import weights



if __name__=='__main__':
	builder = DeepMCR()
	recon = builder.build()


