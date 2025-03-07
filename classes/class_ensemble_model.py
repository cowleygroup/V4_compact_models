# basic structure:
#	for ensemble of models
#	have one base model, and lists of ensemble weights
#	keep swapping out weights, training base model, and update lists
#
# testing model is done in another class (linearmapping_ensemble)
#
# assumes each session will have a different linear mapping.
#	for training, each session will start with a random linear mapping (beta) (even if that session has been seen before)
#		the models are trained for a number of passes on that session's data (fits the shared weights + beta weights)
#		this is then repeated for the next session
#
# other important aspects:
#	- output of each readout network is a K x N x N activation map (the embeddings). These embeddings are then linearly
#		read out to predict the V4 responses. You could relax this with a two-stage linear mapping such as in Bashivan 2019
#		

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

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import DepthwiseConv2D

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import regularizers
from numpy.random import seed

from keras.saving import load_model
# from tensorflow.keras.models import clone_model --- later version than what I have

from sklearn.linear_model import Ridge
from scipy.linalg import eigh
from scipy.linalg import svd
from sklearn.decomposition import PCA
import gc

from numpy import linalg as LA

import pickle
import time


class ModelEnsembleClass: 
 
	def __init__(self, num_members=25, internal_save_folder='./data_ensemble_model/'):
		# internal_save_folder: (folder path), place to store each model weights + optimizers
		#			during training (necessary to keep track of optimizer states)
		#			(internal files)

		self.num_members = num_members

		self.internal_save_folder = internal_save_folder

		self.num_output_vars = 200

		self.num_layers = 4
		self.batch_size = 64
		self.initial_models_weights = None
		# self.momentum = 0.75


	def initialize_model(self, num_layers=4, num_filters=512, num_inside_filters=256, learning_rate=1e0, momentum=0.75):
		# initializes a model (readout network) based on specific hyperparameters
		#	model uses K residual blocks
		#		each residual block goes from M inputs to P inside filters back to M outputs (and input and output are then summed)
		#		idea is that each block does some processing, added on top of input
		#		each block also does some dim reduction (num_inner_filters < num_filters of the output), aka "accordian"
		#
		# INPUT:
		#	num_layers: (integer), number of residual blocks after the first dim-reduction step
		#	num_filters: (integer), number of output filters for each residual block (and output of the network)
		#	num_inside_filters: (integer), number of filters inside the residual block
		#	learning_rate: (float, e.g. 1e-1), learning rate for SGD. momentum = 0.7
		# OUTPUT:
		#	None.

		self.momentum = momentum
		self.batch_size = 64
		self.num_output_vars = 200
		self.num_neurons = 50
		self.num_layers = num_layers

		# learning_rate = learning_rate
		self.num_layers = num_layers
		# num_filters = num_filters  # outer projection of residual block
		# num_inside_filters = num_inside_filters # inner projection of residual block, "dim reduction"

		# self.base_model = self.define_architecture(num_layers, num_filters, num_inside_filters)

		# self.optimizer = SGD(learning_rate=self.learning_rate, momentum=self.momentum, clipvalue=100.)
		# self.base_model.compile(optimizer=self.optimizer, loss='mean_squared_error')

		# self.base_model.summary()

		# self.embedding_model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer('layer{:d}_add'.format(num_layers)).output)
		# self.spatial_features_model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer('layer{:d}_add'.format(self.num_layers)).output)

		# re-initialize networks to form an ensemble
		self.initial_models_weights = []

		for imember in range(self.num_members):
			print('initializing model ' + str(imember) + '...')
			K.clear_session()
			# seed_index = 31490 + imember
			# seed(seed_index)
			# tf.random.set_seed(seed_index)

			seed_index = 31490 + imember
			tf.keras.utils.set_random_seed(seed_index)

			self.base_model = self.define_architecture(num_layers, num_filters, num_inside_filters)
				# re-initializes model with different (random) weights
			self.optimizer = SGD(learning_rate=learning_rate, momentum=self.momentum, clipvalue=0.5)
			# self.optimizer = Adam(learning_rate=learning_rate, clipvalue=0.5)
			self.base_model.compile(optimizer=self.optimizer, loss='mean_squared_error')

			self.base_model.save(self.internal_save_folder + 'internal_ensemble_member{:d}.keras'.format(imember))
				# saves model weights + optimizer states

			self.initial_models_weights.append(self.base_model.get_weights())


	def define_architecture(self, num_layers, num_filters, num_inside_filters):
		# define architecture
		x_input = Input(shape=(14,14,1024), name='feature_input')

		x = Conv2D(filters=num_filters, kernel_size=(1,1), strides=1, padding='same', name='initial_conv_layer')(x_input)
		x = BatchNormalization(axis=-1, name='initial_bn')(x)
		x = Activation(activation='relu', name='initial_act')(x)

		# downsampling step
		x = SeparableConv2D(filters=num_filters, kernel_size=(3,3), strides=2, padding='same', name='initial_sepconv_layer')(x)

		for ilayer in range(num_layers):
			# ResNet-like block
			x_block = SeparableConv2D(filters=num_inside_filters, kernel_size=(3,3), strides=1, padding='same', name='layer{:d}_conv1'.format(ilayer+1))(x)
			x_block = BatchNormalization(axis=-1, name='layer{:d}_bn1'.format(ilayer+1))(x_block)
			x_block = Activation(activation='relu', name='layer{:d}_act1'.format(ilayer+1))(x_block)
			x_block = SeparableConv2D(filters=num_inside_filters, kernel_size=(3,3), strides=1, padding='same', name='layer{:d}_conv2'.format(ilayer+1))(x_block)
			x_block = BatchNormalization(axis=-1, name='layer{:d}_bn2'.format(ilayer+1))(x_block)
			x_block = Activation(activation='relu', name='layer{:d}_act2'.format(ilayer+1))(x_block)
			x_block = Conv2D(filters=num_filters, kernel_size=(1,1), strides=1, padding='same', name='layer{:d}_conv3'.format(ilayer+1))(x_block)
	
			x = Add(name='layer{:d}_add'.format(ilayer+1))([x, x_block])

		x = Conv2D(filters=self.num_output_vars, kernel_size=(1,1), strides=1, padding='same', name='mixing_stage')(x)
		x = DepthwiseConv2D(kernel_size=(7,7), strides=1, padding='valid', name='spatial_pool_stage')(x)

		x = Flatten(name='embeddings')(x)

		base_model = Model(inputs=x_input, outputs=x)  # creates new network with smaller top network

		return base_model


	def set_base_model(self, imember):
		# load imember's weights into base_model

		self.base_model = load_model(self.internal_save_folder + 'internal_ensemble_member{:d}.keras'.format(imember))
				# loads model weights + optimizer state


	def update_model_weights(self, imember):
		# take base_model's weights and store them (updating stored weights)

		self.base_model.save(self.internal_save_folder + 'internal_ensemble_member{:d}.keras'.format(imember))


	def change_learning_rate(self, learning_rate, momentum=None):
		# updates each model's learning rate for their own optimizer

		if momentum is None:
			momentum = self.momentum

		self.optimizer = SGD(learning_rate=learning_rate, momentum=momentum, clipvalue=0.5)
		# self.optimizer = Adam(learning_rate=learning_rate, clipvalue=0.5)

		for imember in range(self.num_members):
			self.set_base_model(imember)
			self.base_model.compile(optimizer=self.optimizer, loss='mean_squared_error')
			self.update_model_weights(imember)


	def reset_twostage_weights(self):
		# resets all twostage (mixing and spatial_pooling) to random, used for training new sessions
		# mixing_stage weights: {(1,1,512,200) (200,)}
		# spatial_pooling weights: {(7,7,200,1), (200,)}

		if self.initial_models_weights is None:  # if you are retraining, initial model weights may not be defined
			self.initial_models_weights = []
			for imember in range(self.num_members):
				self.set_base_model(imember)
				self.initial_models_weights.append(self.base_model.get_weights())

		for imember in range(self.num_members):

			self.set_base_model(imember)
			model_weights = self.base_model.get_weights()

			# intercepts
			model_weights[-1] = np.zeros(model_weights[-1].shape)
			model_weights[-3] = np.zeros(model_weights[-3].shape)

			# weights
			model_weights[-2] = np.zeros(model_weights[-2].shape)  # sets to initial values
			model_weights[-4] = np.zeros(model_weights[-4].shape)
			
			# set to initial (permuted) weights
			rng = np.random.default_rng()
			model_weights[-2][:,:,:self.num_neurons,:] = 0.001 * rng.permuted(np.copy(self.initial_models_weights[imember][-2]))[:,:,:self.num_neurons,:]  # sets to initial values
			model_weights[-4][:,:,:,:self.num_neurons] = 0.001 * rng.permuted(np.copy(self.initial_models_weights[imember][-4]))[:,:,:,:self.num_neurons]

			self.base_model.set_weights(model_weights)

			self.update_model_weights(imember)

			# # weights
			# s = self.models_weights[imember][-2].shape
			# self.models_weights[imember][-2][:,:,:self.num_neurons,:] = 0.01 * np.random.standard_normal(size=(s[0],s[1],self.num_neurons,s[3]))
			# s = self.models_weights[imember][-4].shape
			# self.models_weights[imember][-4][:,:,:,:self.num_neurons] = 0.01 * np.random.standard_normal(size=(s[0],s[1],s[2],self.num_neurons))


	def train_models(self, features_train, responses_train, num_passes_per_set, reset_weights_flag=True):
		## DOCUMENT

		num_samples = responses_train.shape[1]

		self.num_neurons = responses_train.shape[0]

		if reset_weights_flag == True:
			self.reset_twostage_weights()  # resets weights for all members

		# set extraneous readout weights to 0
		weights = self.base_model.get_weights()
		weights[-1][self.num_neurons:] = 0.
		weights[-3][self.num_neurons:] = 0.
		weights[-2][:,:,self.num_neurons:,:] = 0.
		weights[-4][:,:,:,self.num_neurons:] = 0.

		# add padding to responses for unfilled units
		responses_train = np.vstack((responses_train, np.zeros(shape=(self.num_output_vars-self.num_neurons,num_samples))))
		
		# train models
		for imember in np.arange(self.num_members):

			# swap model weights
			self.set_base_model(imember)

			self.base_model.fit(features_train, responses_train.T, epochs=num_passes_per_set, batch_size=self.batch_size, shuffle=True, verbose=0)

			# update weights
			self.update_model_weights(imember)


	def get_predicted_ensemble_responses(self, features, removePadding=True):
		# averages responses over ensemble

		num_images = features.shape[0]
		responses_avg = np.zeros((self.num_output_vars, num_images))
		for imember in range(self.num_members):

			self.set_base_model(imember)

			for ibatch_start in range(0,features.shape[0],self.batch_size):
				responses_avg[:,ibatch_start:ibatch_start+self.batch_size] += self.base_model(features[ibatch_start:ibatch_start+self.batch_size], training=False).numpy().T

		# remove padding
		if removePadding == True:
			responses_avg = responses_avg[:self.num_neurons,:]

		return responses_avg / self.num_members


	def get_predicted_responses_from_ith_member(self, features, imember, remove_padding=True):
		# returns predicted responses from ith_member
		#	useful for computing model uncertainty
		#
		# INPUT:
		#	features: (num_images, num_pixels, num_pixels, num_filters), ResNet50 activation maps
		#	imember: (integer), ith member index
		#	remove_padding: (boolean), if True: removes zero padding from non-used neuron weights (model has 300 output variables)
		# OUTPUT:
		#	responses: (num_neurons, num_images), predicted responses

		self.set_base_model(imember)

		num_images = features.shape[0]
		responses = np.zeros((self.num_output_vars, num_images))

		for ibatch_start in range(0,features.shape[0],self.batch_size):
			responses[:,ibatch_start:ibatch_start+self.batch_size] = self.base_model(features[ibatch_start:ibatch_start+self.batch_size], training=False).numpy().T

		# remove padding
		if remove_padding == True:
			responses = responses[:self.num_neurons,:]

		return responses


	def set_Beta_weights(self, W_S_list, W_M_list):
		# DEPRECATED
		#
		# sets Beta weights (spatial pooling + mixing weights) to desired values
		#
		# INPUT:
		#	W_S_list: (list of length num_members), where W_S_list[imember] is (num_pixels, num_pixels, num_neurons)  spatial pooling weights
		#	W_M_list: (list of length num_members), where W_M_list[imember] is (num_filters, num_neurons)  mixing weights
		#
		# OUTPUT:
		#	None.
		
		num_pixels = W_S_list[0].shape[0]
		self.num_neurons = W_S_list[0].shape[2]
		num_filters = W_M_list[0].shape[0]

		# perform linear regression
		for imember in range(self.num_members):

			W_S = W_S_list[imember]
			W_M = W_M_list[imember]

			# set intercepts to zero
			self.models_weights[imember][-1] = np.zeros(self.models_weights[imember][-1].shape) 
			self.models_weights[imember][-3] = np.zeros(self.models_weights[imember][-3].shape)

			# first set to zero (for extra output variables)
			self.models_weights[imember][-2] = np.zeros((self.models_weights[imember][-2].shape))
			self.models_weights[imember][-4] = np.zeros((self.models_weights[imember][-4].shape))

			# now set weights recovered by alternating procedure
			self.models_weights[imember][-2][:,:,:self.num_neurons,0] = W_S
			self.models_weights[imember][-4][0,0,:,:self.num_neurons] = W_M

		return None


	def get_embeddings_for_ith_member(self, features, imember, flatten_flag=False):
		# returns embeddings of readout network for given member
		#
		# INPUT:
		#	features: (num_images, num_pixels, num_pixels, num_filters), features from pre-trained DNN (ResNet50)
		#	imember: (integer from 0 to num_members-1), indexes which ensemble member (a readout network) to query
		#	flatten_flag: (boolean), flattens embeddings across pixels and filters
		#
		# OUTPUT:
		#	embeddings: (num_images, num_pixels, num_pixels, num_filters), output embeddings of model to use to linearly predict V4 responses
		#		if flatten_flag == True: (num_embed_vars, num_images)

		self.set_base_model(imember)
		embedding_model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer('layer{:d}_add'.format(self.num_layers)).output)

		embeddings = embedding_model.predict(features, verbose=False)

		if flatten_flag == True:
			embeddings = np.reshape(embeddings, (embeddings.shape[0],-1)).T

		return embeddings


	def save_models(self, filetag='base_model', save_folder=None):
		# stores ensemble's weights for later use, including all Beta weights
		#
		# INPUT:
		#	filetag: (string), filename not including path or .h5, it will be appended with "_memberXX.keras"
		#	save_folder: (string), filepath (e.g., '/jukebox/pillow/bcowley/saved_models/')
		#
		# OUTPUT:
		#	None.
		#
		# Notes: does not save embedding_model, as that can be recovered from full model

		if save_folder == None:
			save_folder = self.internal_save_folder

		for imember in range(self.num_members):
			self.set_base_model(imember)
			self.base_model.save(save_folder + filetag + '_member{:d}.keras'.format(imember))


	def load_models(self, filetag='ensemble_model', load_folder=None, forTraining=False):
		# loads ensemble's weights + beta weights
		#	
		# INPUT:
		#	filetag: (string), filename not including path or .h5
		#	load_folder: (string), filepath (e.g., '/jukebox/pillow/bcowley/saved_models/')
		#	forTraining: (boolean), if True: recompiles base_model to be re-trained again (e.g., so you can change step size between sessions, etc.)
		#
		# OUTPUT:
		#	None.

		if load_folder == None:
			load_folder = self.internal_save_folder

		for imember in range(self.num_members):
			self.base_model = load_model(load_folder + filetag + '_member{:d}.keras'.format(imember))

			model_weights = self.base_model.get_weights()

			# intercepts
			model_weights[-1] = np.zeros(model_weights[-1].shape)
			model_weights[-3] = np.zeros(model_weights[-3].shape)

			# weights
			model_weights[-2] = np.zeros(model_weights[-2].shape)  # sets to initial values
			model_weights[-4] = np.zeros(model_weights[-4].shape)

			# weights
			model_weights[-2] = 0.01*np.random.standard_normal(size=model_weights[-2].shape)  # sets to initial values
			model_weights[-4] = 0.01*np.random.standard_normal(size=model_weights[-4].shape)

			self.base_model.set_weights(model_weights)

			self.update_model_weights(imember)  # saves into internal models

		# identify number of neurons from Beta
		if True:
			s = np.sum(np.abs(self.base_model.get_weights()[-2][:,:,:,0]), axis=(0,1)) # (num_neurons,), sums up |weights|
			self.num_neurons = np.sum(s > 1e-8)   # look at spatial pooling weights, sum them up, and see if weights are > 0


