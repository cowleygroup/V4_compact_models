# compact model class
#
#	shared model for training directly on V4 data
#
#   the trick to add:
#	different sessions have different number of V4 neurons, so have a large number (200) and don't use extra
#	and set the linear mapping Betas each session

# Written by B. Cowley, 2025


import numpy as np
import tensorflow as tf

import sys

gpu_device = sys.argv[1]
print('using gpu ' + gpu_device)
import os
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_device

from tensorflow.keras import backend as K
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.keras.utils.disable_interactive_logging()


from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import Flatten, Dense

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model

from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

from numpy import linalg as LA
from scipy import ndimage

import pickle
import time


class DistilledModelClass: 
	# class that defines the "distilled/pruned model"

	def __init__(self):

		self.num_layers = 5
		self.batch_size = 64


	def initialize_model(self, num_layers=5, num_filters_earlylayers=100, num_filters_latelayers=100, num_passes_per_set=1, learning_rate=2e0):
		# initializes compact model
		#
		# INPUT:
		#	num_layers: (integer >= 3), determines number of total layers (includes first rgb conv layer)
		# OUTPUT:
		#	none.

		if num_layers < 3:
			raise ValueError('num_layers needs to be 3 or greater due to striding')

		# model hyperparameters
		self.num_output_neurons = 200  # way more than any one session; will be padding
		self.num_layers = num_layers
		self.num_passes_per_set = num_passes_per_set
		
		# define architecture
		x_input = Input(shape=(112,112,3), name='image_input')
		x = x_input

		# first layer: 5x5 conv2d (not separable) --- to allow a lot of flexibility
		ilayer = 0
		x = Conv2D(filters=num_filters_earlylayers, kernel_size=(5,5), padding='same', name='layer{:d}_conv'.format(ilayer))(x)
		x = BatchNormalization(axis=-1, name='layer{:d}_bn'.format(ilayer))(x)
		x = Activation(activation='relu', name='layer{:d}_act'.format(ilayer))(x)

		for ilayer in range(1, num_layers):
			if ilayer < 3:
				stride_length=2
			else:
				stride_length=1

			if ilayer <= 2:
				num_filters = num_filters_earlylayers
			else:
				num_filters = num_filters_latelayers

			x = SeparableConv2D(filters=num_filters, kernel_size=(5,5), strides=stride_length, padding='same', name='layer{:d}_conv'.format(ilayer))(x)
			x = BatchNormalization(axis=-1, name='layer{:d}_bn'.format(ilayer))(x)
			x = Activation(activation='relu', name='layer{:d}_act'.format(ilayer))(x)

		# output
		x = Conv2D(filters=self.num_output_neurons, kernel_size=(1,1), strides=1, padding='same', name='mixing_stage')(x)
		x = DepthwiseConv2D(kernel_size=(28,28), strides=1, padding='valid', name='spatial_pool_stage')(x)
		x = Flatten(name='output_responses')(x)

		self.model = Model(inputs=x_input, outputs=x)

		# optimizer = Adam(learning_rate=1e-4)
		optimizer = SGD(learning_rate=learning_rate, momentum=0.7, clipvalue=0.5)

		self.model.compile(optimizer=optimizer, loss='mean_squared_error')

		self.weights_initial = self.model.get_weights()


	def train_model(self, images_recentered_train, responses_train):
		# trains compact model
		#
		# INPUT:
		#	images_recentered_train: (num_images, num_pixels, num_pixels, 3), training images, 112x112, already re-centered
		#	responses_train: (num_neurons,num_images), training responses
		#
		# OUTPUT:
		#	none. updates model's weights in place.

		num_neurons = responses_train.shape[0]
		num_samples = responses_train.shape[1]

		# add padding to responses for unfilled units
		responses_train = np.vstack((responses_train, np.zeros(shape=(self.num_output_neurons-num_neurons,num_samples))))

		self.model.fit(images_recentered_train, responses_train.T, epochs=self.num_passes_per_set, shuffle=True, verbose=0)


	def smooth_weights(self):
		# smooths the kernel weights for layers 1,...,num_layers (including last Beta readout mapping)
		#  uses sigma=0.5, found to be a reasonable amount of smoothing (any more and performance begins to drop)
		#  typically you want to smooth after each pass through one large zip
		#
		# updates weights in model in-place.

		sigma = 0.5  # smoothing constant for gaussian filter

		inds_weights_each_layer = [0,6,13,20,27,36]

		weights = self.model.get_weights()

		for ilayer in range(1,self.num_layers+1):  # does not smooth the first rgb-conv layer (hurts performance)
												   # does include last spatial readout layer (36)
			ind = inds_weights_each_layer[ilayer]
			w = weights[ind]
			
			w = ndimage.gaussian_filter(w, sigma=[sigma,sigma,0.,0.])

			weights[ind] = w

		self.model.set_weights(weights)


	def reset_readout_weights(self, num_neurons):
		# resets all twostage (mixing and spatial_pooling) to random, used for training new sessions
		# mixing_stage weights: {(1,1,512,300) (300,)}
		# spatial_pooling weights: {(7,7,300,1), (300,)}

		weights = self.model.get_weights()

		# intercepts
		weights[-1] = np.zeros(weights[-1].shape)
		weights[-3] = np.zeros(weights[-3].shape)

		# kernel weights
		weights[-2] = np.zeros(weights[-2].shape)
		weights[-4] = np.zeros(weights[-4].shape)

		weights[-2][:,:,:num_neurons,:] = self.weights_initial[-2][:,:,:num_neurons,:]
		weights[-4][:,:,:,:num_neurons] = self.weights_initial[-4][:,:,:,:num_neurons]

		self.model.set_weights(weights)


	def get_embeddings(self, images_recentered):
		# returns embeddings before factorized linear mapping
		#
		# INPUT:
		#	images_recentered: (num_images, num_pixels, num_pixels, 3), images already re-centered, 112 x 112
		#
		# OUTPUT:
		#	embeddings: (num_feature_vars,num_images), embeddings for this distilled model (before factorized linear mapping)

		return self.get_activity_from_layer(images_recentered, ilayer=4)


	def get_predicted_responses(self, images_recentered):
		# returns predicted responses
		#
		# INPUT:
		#	images_recentered: (num_images, num_pixels, num_pixels, 3), images already re-centered, 112 x 112
		#
		# OUTPUT:
		#	responses: (num_neurons,num_images), predicted responses for this compact model

		responses = []
		batch_size = 64
		num_images = images_recentered.shape[0]
		for ibatch in range(0,num_images,batch_size):
			responses.append(self.model(images_recentered[ibatch:ibatch+batch_size]))

		return np.concatenate(responses,axis=0).T
			# (num_neurons, num_images)


	def get_activity_from_layer(self, images_recentered, ilayer):
		# returns activity from layer ilayer to given images
		#  activity is the "output" of the layer --> conv + readout + batchnorm + relu
		#
		# INPUT:
		#	images_recentered: (num_images, num_pixels, num_pixels, 3), images already re-centered, 112 x 112
		#	ilayer: (integer between 0 and 4), denotes the ith layer to get the activity
		#
		# OUTPUT:
		#	activity: (num_images, num_pixels, num_pixels, num_filters): activity from the ith layer

		model_activity = Model(inputs=self.model.inputs, outputs=self.model.get_layer('layer{:d}_act'.format(ilayer)).output)

		return model_activity.predict(images_recentered)


	def get_activity_from_convolutional_filters_layer(self, images_recentered, ilayer):
		# returns activity from layer ilayer to given images
		# where applicable, activity is from output of convolutional filter (inside the separable convolution)
		#
		# INPUT:
		#	images_recentered: (num_images, num_pixels, num_pixels, 3), images already re-centered, 112 x 112
		#	ilayer: (integer between 0 and 5), denotes the ith layer to get the activity
		#		ilayer = 5 is the output of the spatial readouts (not summed)
		#
		# OUTPUT:
		#	activity: (num_images, num_pixels, num_pixels, num_filters): activity from the ith layer

		if ilayer == 0:  # no separable conv, so output activity
			model_activity = Model(inputs=self.model.inputs, outputs=self.model.get_layer('layer{:d}_act'.format(ilayer)).output)
		elif ilayer >= 1 and ilayer <= 4:
			if ilayer < 3:
				stride_length = 2
			else:
				stride_length = 1

			x = DepthwiseConv2D(kernel_size=(5,5), strides=stride_length, padding='same', name='next_conv')(self.model.get_layer('layer{:d}_act'.format(ilayer-1)).output)
			model_activity = Model(inputs=self.model.inputs, outputs=x)

			weights = self.model.get_layer('layer{:d}_conv'.format(ilayer)).get_weights()
			model_activity.get_layer('next_conv').set_weights([weights[0], np.zeros((weights[0].shape[2],))])
		else:  # spatial readout, ilayer = 5
			x = DepthwiseConv2D(kernel_size=(28,28), strides=1, padding='valid', name='spatial_readout')(self.model.get_layer('layer4_act').output)
			model_activity = Model(inputs=self.model.inputs, outputs=x)

			weights = self.model.get_layer('Beta').get_weights()
			w = weights[0]
			w = np.reshape(w, (28,28,-1))
			w = w[:,:,:,np.newaxis]
			model_activity.get_layer('spatial_readout').set_weights([w, np.zeros((w.shape[2],))])

		return model_activity.predict(images_recentered)


	def get_number_of_mixing_filters_per_layer(self):
		# returns number of kernel filters per layer
		#
		# INPUT:
		#	none.
		# OUTPUT:
		#	nums_filters: (num_layers,), number of filters per layer

		weights = self.model.get_weights()
		inds_weights_each_layer = [0,7,14,21,28]
									# gives number of mixing (readout) filters per layer

		nums_filters = np.zeros((self.num_layers,))
		for ilayer in range(self.num_layers):
			nums_filters[ilayer] = weights[inds_weights_each_layer[ilayer]].shape[-1]

		return nums_filters.astype('int')


	def get_number_of_kernel_filters_per_layer(self):
		# returns number of kernel filters per layer
		#
		# INPUT:
		#	none.
		# OUTPUT:
		#	nums_filters: (num_layers,), number of filters per layer

		weights = self.model.get_weights()
		inds_weights_each_layer = [0,0,7,14,21,28] # looks weird, but needs special indexing
									# gives number of kernel filters per layer

		nums_filters = np.zeros((self.num_layers+1,))
		for ilayer in range(self.num_layers+1):
			nums_filters[ilayer] = weights[inds_weights_each_layer[ilayer]].shape[-1]

		return nums_filters


	def get_kernel_weights_per_layer(self):
		# returns kernel weights per layer (6 in total, including spatial readout)
		#
		# INPUT:
		#	none.
		#
		# OUTPUT:
		#	weights: (list of length num_layers), where weights[ilayer] is 
		#		(P,P,3,K) for first layer (P -- kernel size=5, K -- num_of_filters)
		#		(P,P,K,1) for layers 1-4 (P -- kernel size=5, K -- num_of_filters)
		#		(P,P,K) for spatial readout (P -- kernel_size=28, K -- num_of filters)

		inds_weights_each_layer = [0,6,13,20,27,34]  #[0,6,13,20,27,34,41,48]  indexes conv layer

		kernel_weights = []
		weights = self.model.get_weights()
		
		kernel_weights.append(weights[0])  # first layer

		for ilayer in range(1,self.num_layers):  

			ind = inds_weights_each_layer[ilayer]

			kernel_weights.append(weights[ind])

		# spatial readout layer
		ind_layer = self.num_layers
		w = weights[inds_weights_each_layer[ind_layer]]
		w = np.reshape(w, (28,28,-1))

		kernel_weights.append(w)

		return kernel_weights


	def set_kernel_weights_for_layer(self, ilayer, weights_new):
		# sets kernel weights for each layer
		#
		# INPUT:
		#	ilayer: (integer, 0 to 5), indicates which layer
		#		ilayer=0 --> initial conv
		#		ilayer=1..4 --> middle convolutional layer
		#		ilayer=5 --> spatial readout layer
		#	weights_new: (list of length num_layers), where weights[ilayer] is 
		#		ilayer=0: (P,P,3,K) for first layer (P -- kernel size=5, K -- num_of_filters)
		#		ilayer=1...4: (P,P,K,1) for layers 1-4 (P -- kernel size=5, K -- num_of_filters)
		#		ilayer=5: (P,P,K) for spatial readout (P -- kernel_size=28, K -- num_of filters)
		#
		# OUTPUT:
		#	none.

		inds_weights_each_layer = [0,6,13,20,27,34]  #[0,6,13,20,27,34,41,48]  indexes conv layer

		kernel_weights = []
		weights = self.model.get_weights()  # list
		
		ind = inds_weights_each_layer[ilayer]

		if ilayer < 5:
			weights[ind] = np.copy(weights_new)
		elif ilayer == 5: # spatial readout layer
			weights[ind] = np.reshape(np.copy(weights_new), (-1,1))

		self.model.set_weights(weights)

		return None


	def get_readout_weights_per_layer(self):
		# returns kernel weights per layer (6 in total, including spatial readout)
		#
		# INPUT:
		#	none.
		#
		# OUTPUT:
		#	weights: (list of length num_layers), where weights[ilayer] is 
		#		(P,P,3,K_out) for first layer (P -- kernel size=5, K_out -- num_of_filters)
		#		(1,1,K_in,K_out) for layers 1-4 (K_in -- num_of_filters for previous layer, K_out -- num of output filters)
		#		(P,P,K_out) for spatial readout (P -- kernel_size=28, K_out -- num_of filters)

		inds_weights_each_layer = [0,7,14,21,28,34]  #[0,6,13,20,27,34,41,48]  indexes conv layer

		readout_weights = []
		weights = self.model.get_weights()
		
		readout_weights.append(weights[0])  # first layer

		for ilayer in range(1,self.num_layers):  

			ind = inds_weights_each_layer[ilayer]

			readout_weights.append(weights[ind])

		# spatial readout layer
		ind_layer = self.num_layers
		w = weights[inds_weights_each_layer[ind_layer]]
		w = np.reshape(w, (28,28,-1))

		readout_weights.append(w)

		return readout_weights


	def save_model(self, filetag='shared_compact_model', save_folder=None):
		# stores model for later use
		#
		# INPUT:
		#	filetage: (string), file name of the saved model, function will append a '.h5' to it
		#	save_folder: (string), where to save the model. if None, saves to results folder.
		#			if included, make sure string ends in '/'
		# OUTPUT:
		#	none. saving function.

		if save_folder == None:
			save_folder = '/usr/people/bcowley/adroit/code/paper_neural_distill/manuscript/fig3/train_shared_network/results/saved_models/'

		self.model.save(save_folder + filetag + '.h5')


	def load_model(self, filetag='shared_compact_model', load_folder=None):
		# loads model
		#
		# INPUT:
		#	filetage: (string), file name of the desired model, function will append a '.h5' to it
		#	load_folder: (string), where to load the model. if None, loads from results folder.
		#			if included, make sure string ends in '/'
		# OUTPUT:
		#	none. loading function.

		if load_folder == None:
			load_folder = '/usr/people/bcowley/adroit/code/paper_neural_distill/manuscript/fig3/train_shared_network/results/saved_models/'

		self.model = load_model(load_folder + filetag + '.h5', custom_objects={'Functional':tf.keras.models.Model})

 
