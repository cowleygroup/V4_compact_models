
# maps embeddings to neural responses
#
# there are a lot of ways to do this:
#	- ridge regression
#	- two-stage mapping, Keras implementation (Klindt 2017)
#	- two-stage mapping, alternating
#	- two-stage mapping, number of factors

# Written by B. Cowley, 2025

import numpy as np
import tensorflow as tf

import sys
import time

gpu_device = sys.argv[1]
print('using gpu ' + gpu_device)
import os
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_device

from tensorflow.keras import backend as K
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.keras.utils.disable_interactive_logging()

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2, l1

from scipy import ndimage
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import ndimage


class LinearMappingClass:


	def __init__(self, lambda_mixing=0.01, lambda_spatialpool=0.01, num_epochs_total=50, learning_rate=0.1, alpha_ridge=1., svd_factorize_flag=False):
		# hyperparameters (for tensorflow implementation)
		self.lambda_mixing = lambda_mixing
		self.lambda_spatialpool = lambda_spatialpool
		self.num_epochs_total = num_epochs_total
		self.learning_rate = learning_rate
		self.alpha_ridge = alpha_ridge
		self.svd_factorize_flag = svd_factorize_flag


	def perform_cross_validation(self, features, raw_responses, metric='r2_ER'):
		# computes 4-fold cross-validation on features to predict responses
		#	only uses tensorflow mapping
		#   only computes R2 after self.num_epochs_total epochs are completed
		#
		#
		# INPUT:
		#	features: (num_images, num_pixels, num_pixels, num_filters), features/embeddings/activations; typically output from a DNN layer
		#	raw_responses: (num_neurons, num_images, num_repeats), raw response spike counts
		#	metric: ('r2_ER' or 'brainscore' or 'both'), computes noise-corrected R2 in different ways
		#				if both, returns both as a list
		#	
		# OUTPUT:
		#	R2s_over_neurons: (num_neurons,), cross-validated, noise-corrected R2s
		#

		num_images = features.shape[0]
		num_folds = 4

		num_test_images_per_fold = np.floor(num_images / num_folds).astype('int')

		# shuffle features + responses
		features = np.copy(features)
		responses = np.copy(raw_responses)

		# shuffle training set
		r = np.random.permutation(num_images)
		features = features[r,:,:,:]
		responses = responses[:,r,:]

		responses_train = np.nanmean(responses, axis=2)  # (num_neurons, num_images)
		responses_hat = np.zeros(responses_train.shape)

		num_pixels = features.shape[1]
		num_filters = features.shape[-1]
		num_neurons = responses.shape[0]

		## train two-stage mapping and compute performance every 10th epoch
		
		# train model 
		if True:
			responses_hat = np.zeros((num_neurons,num_images)) * np.nan
			for ifold in range(num_folds):
				model = self.initialize_factorized_linear_mapping(num_pixels, num_filters, num_neurons)

				inds_test = np.arange(ifold*num_test_images_per_fold, (ifold+1)*num_test_images_per_fold)
				inds_train =  np.array([x for x in range(num_folds * num_test_images_per_fold) if x not in inds_test])

				Xtrain = features[inds_train,:,:,:]
				Xtest = features[inds_test,:,:,:]
				Ytrain = responses_train[:,inds_train]
				num_train_images = Xtrain.shape[0]

				weights = model.get_weights()
				weights[-1] = np.mean(Ytrain,axis=1)
				model.set_weights(weights)

				for iepoch in range(self.num_epochs_total):
					batch_size = 64
					model.fit(Xtrain, Ytrain.T, epochs=10, batch_size=batch_size, shuffle=True, verbose=0)
					print('fold {:d} epoch {:d}'.format(ifold,iepoch))

				responses_hat[:,inds_test] = model.predict(Xtest, verbose=0).T

			if metric == 'r2_ER':
				R2s_over_neurons = self.compute_r2_ER(responses[:,:-4,:], responses_hat[:,:-4])
				return R2s_over_neurons
			elif metric == 'brainscore':
				R2s_over_neurons = self.compute_r2_ER(responses[:,:-4,:], responses_hat[:,:-4])
				return R2s_over_neurons
			elif metric == 'both':
				R2s_brainscore = self.compute_brain_score(responses[:,:-4,:], responses_hat[:,:-4])
				R2s_biascorrected = self.compute_r2_ER(responses[:,:-4,:], responses_hat[:,:-4])

				return R2s_brainscore, R2s_biascorrected


	def perform_one_fold_with_early_stopping(self, features, raw_responses, metric='r2_ER', verbose=True):
		#	-- basically, choose some number of epochs and keep the same across models
		#		(you can vary lambdas instead)
		# computes 1 fold of 8-fold cross-validation on features to predict responses
		#		(helpful for hyperparam optimization)
		#	only uses tensorflow mapping
		#   computes performance at each epoch
		#
		# INPUT:
		#	features: (num_images, num_pixels, num_pixels, num_filters), features/embeddings/activations; typically output from a DNN layer
		#	raw_responses: (num_neurons, num_images, num_repeats), raw response spike counts
		#	metric: ('r2_ER' or 'brainscore' or 'both'), computes noise-corrected R2 in different ways
		#				if both, returns both as a list
		#	
		# OUTPUT:
		#	noise_corrected_R2s: (num_neurons,), cross-validated, noise-corrected R2s
		#

		num_images = features.shape[0]
		num_folds = 8
		num_neurons = raw_responses.shape[0]

		num_test_images_per_fold = np.floor(num_images / num_folds).astype('int')
		num_val_images_per_fold = np.floor(num_images / num_folds).astype('int')

		# shuffle features + responses
		features = np.copy(features)
		responses = np.copy(raw_responses)

		# # shuffle training set
		# r = np.random.permutation(num_images)
		# features = features[r,:,:,:]
		# responses = responses[:,r,:]

		responses_train = np.nanmean(responses, axis=2)  # (num_neurons, num_images)

		num_pixels = features.shape[1]
		num_filters = features.shape[-1]
		num_neurons = responses.shape[0]

		## train two-stage mapping and compute performance every 10th epoch
		
		# train model 

		ifold = 1  # only one fold
		model = self.initialize_factorized_linear_mapping(num_pixels, num_filters, num_neurons)

		inds_test = np.arange(ifold*num_test_images_per_fold, (ifold+1)*num_test_images_per_fold)
		inds_val = np.arange(num_images - (ifold+1)*num_val_images_per_fold, num_images - ifold*num_val_images_per_fold)
		inds_train =  np.array([x for x in range(num_folds * num_test_images_per_fold) if x not in inds_test and x not in inds_val])

		Xtrain = features[inds_train,:,:,:]
		Xtest = features[inds_test,:,:,:]
		Xval = features[inds_val,:,:,:]
		Ytrain = responses_train[:,inds_train]
		num_train_images = Xtrain.shape[0]

		# initialize weights 
		weights = model.get_weights()
		weights[-1] = np.mean(Ytrain,axis=1)
		model.set_weights(weights)

		if self.svd_factorize_flag == True:
			self.initialize_with_svd_Betas(model, Xtrain, Ytrain)
				# updates weights in place

		R2s_train_over_epochs = np.zeros((self.num_epochs_total,));
		R2s_val_over_epochs = np.zeros((self.num_epochs_total,));
		R2s_test_over_epochs = np.zeros((self.num_epochs_total,));

		for iepoch in range(self.num_epochs_total):
			weights_before = model.get_weights()

			batch_size = 64
			model.fit(Xtrain, Ytrain.T, epochs=10, batch_size=batch_size, shuffle=True, verbose=0)

			weights_after = model.get_weights()

			diff_max = 0
			norm = 0
			for ilayer in range(len(weights_before)):
				diffs = weights_after[ilayer] - weights_before[ilayer]
				if diff_max < np.max(np.abs(diffs)):
					diff_max = np.max(np.abs(diffs))

				norm += np.sum(diffs**2)

			responses_hat_train = model.predict(Xtrain, verbose=0).T
			responses_hat_val = model.predict(Xval, verbose=0).T
			responses_hat_test = model.predict(Xtest, verbose=0).T

			R2_train = np.nanmedian(self.compute_r2_ER(responses[:,inds_train,:], responses_hat_train))
			R2_val = np.nanmedian(self.compute_r2_ER(responses[:,inds_val,:], responses_hat_val))
			R2_test = np.nanmedian(self.compute_r2_ER(responses[:,inds_test,:], responses_hat_test))

			if verbose:
				norm = np.sqrt(norm)
				print('fold {:d} epoch {:d}, R2_train = {:f}, R2_val = {:f}, R2_test = {:f}, diff_clipvalue = {:f}, diff_norm = {:f}'.format(ifold, iepoch, R2_train, R2_val, R2_test, diff_max, norm))

			R2s_train_over_epochs[iepoch] = R2_train
			R2s_val_over_epochs[iepoch] = R2_val
			R2s_test_over_epochs[iepoch] = R2_test

		iepoch_max = np.argmax(R2s_val_over_epochs)

		return R2s_test_over_epochs[iepoch_max]



	def perform_cross_validation_with_early_stopping(self, features, raw_responses, metric='r2_ER', verbose=True):
		# perform 8 fold nested cross-val (where 1/8 is val, 1/8 is test for each fold)
		#	-- basically, choose some number of epochs and keep the same across models
		#		(you can vary lambdas instead)
		# computes 4-fold cross-validation on features to predict responses
		#	only uses tensorflow mapping
		#   computes performance at each epoch
		#
		# INPUT:
		#	features: (num_images, num_pixels, num_pixels, num_filters), features/embeddings/activations; typically output from a DNN layer
		#	raw_responses: (num_neurons, num_images, num_repeats), raw response spike counts
		#	metric: ('r2_ER' or 'brainscore' or 'both'), computes noise-corrected R2 in different ways
		#				if both, returns both as a list
		#	
		# OUTPUT:
		#	noise_corrected_R2s: (num_neurons,), cross-validated, noise-corrected R2s
		#

		num_images = features.shape[0]
		num_folds = 8
		num_neurons = raw_responses.shape[0]

		num_test_images_per_fold = np.floor(num_images / num_folds).astype('int')
		num_val_images_per_fold = np.floor(num_images / num_folds).astype('int')

		# shuffle features + responses
		features = np.copy(features)
		responses = np.copy(raw_responses)

		# shuffle training set
		r = np.random.permutation(num_images)
		features = features[r,:,:,:]
		responses = responses[:,r,:]

		responses_train = np.nanmean(responses, axis=2)  # (num_neurons, num_images)

		num_pixels = features.shape[1]
		num_filters = features.shape[-1]
		num_neurons = responses.shape[0]

		## train two-stage mapping and compute performance every 10th epoch
		
		# train model 
		if True:
			responses_hat_val = np.zeros((self.num_epochs_total,num_neurons,num_images))
			responses_hat_test = np.zeros((self.num_epochs_total,num_neurons,num_images))
			for ifold in range(num_folds):
				model = self.initialize_factorized_linear_mapping(num_pixels, num_filters, num_neurons)

				inds_test = np.arange(ifold*num_test_images_per_fold, (ifold+1)*num_test_images_per_fold)
				inds_val = np.arange(num_images - (ifold+1)*num_val_images_per_fold, num_images - ifold*num_val_images_per_fold)
				inds_train =  np.array([x for x in range(num_folds * num_test_images_per_fold) if x not in inds_test and x not in inds_val])

				Xtrain = features[inds_train,:,:,:]
				Xtest = features[inds_test,:,:,:]
				Xval = features[inds_val,:,:,:]
				Ytrain = responses_train[:,inds_train]
				num_train_images = Xtrain.shape[0]

				# initialize weights 
				weights = model.get_weights()
				weights[-1] = np.mean(Ytrain,axis=1)
				model.set_weights(weights)

				if self.svd_factorize_flag == True:
					self.initialize_with_svd_Betas(model, Xtrain, Ytrain)

				for iepoch in range(self.num_epochs_total):
					batch_size = 64
					model.fit(Xtrain, Ytrain.T, epochs=10, batch_size=batch_size, shuffle=True, verbose=0)

					# responses_hat_val[iepoch][:,inds_val] = model.predict(Xval, verbose=0).T
					# responses_hat_test[iepoch][:,inds_test] = model.predict(Xtest, verbose=0).T

					responses_hat_val[iepoch][:,inds_val] = self.get_model_predicted_responses(model, Xval)
					responses_hat_test[iepoch][:,inds_test] = self.get_model_predicted_responses(model, Xtest)

					if verbose:
						print('fold {:d} epoch {:d}'.format(ifold, iepoch))

			# choose best epoch with validation data
			R2s_val_over_epochs = np.zeros((self.num_epochs_total,))
			for iepoch in range(self.num_epochs_total):
				R2s_val_over_epochs[iepoch] = np.mean(self.compute_r2_ER(responses[:,:-4,:], responses_hat_val[iepoch][:,:-4]))

			iepoch_max = np.nanargmax(R2s_val_over_epochs)

			# now report R2 test
			if metric == 'r2_ER':
				R2s_over_neurons = self.compute_r2_ER(responses[:,:-4,:], responses_hat_test[iepoch_max][:,:-4])
				return R2s_over_neurons
			elif metric == 'brainscore':
				R2s_over_neurons = self.compute_brain_score(responses[:,:-4,:], responses_hat_test[iepoch_max][:,:-4])
				return R2s_over_neurons
			elif metric == 'both':
				R2s_brainscore = self.compute_brain_score(responses[:,:-4,:], responses_hat_test[iepoch_max][:,:-4])
				R2s_biascorrected = self.compute_r2_ER(responses[:,:-4,:], responses_hat_test[iepoch_max][:,:-4])

				return R2s_brainscore, R2s_biascorrected
			elif metric == 'responses':
				R2s_biascorrected = self.compute_r2_ER(responses[:,:-4,:], responses_hat_test[iepoch_max][:,:-4])
				return R2s_biascorrected, responses[:,:-4,:], responses_hat_test[iepoch_max][:,:-4]


	def train_with_early_stopping(self, features_train, responses_train, features_val, responses_raw_val, features_test, responses_raw_test, verbose=True):
		# train one instance of factorized linear mapping given training, validation, and test data (user splits this up)
		#
		# INPUT:
		#	features_train: (num_images_train, num_pixels, num_pixels, num_filters), features/embeddings/activations for training; typically output from a DNN layer
		#	responses_train:
		#	features_val: (num_images_val, num_pixels, num_pixels, num_filters), DNN features for validation (choosing hyperparams, e.g., early stopping)
		#	responses_raw_val:
		#	features_test: (num_images_test, num_pixels, num_pixels, num_filters)
		#	responses_raw_test:
		# raw_responses: (num_neurons, num_images, num_repeats), raw response spike counts
		#	
		# OUTPUT:
		#	noise_corrected_R2s: (num_neurons,), cross-validated, noise-corrected R2s
		#

		num_images_train = features_train.shape[0]
		num_images_val = features_val.shape[0]
		num_images_test = features_test.shape[0]

		num_neurons = responses_train.shape[0]

		num_pixels = features_train.shape[1]
		num_filters = features_train.shape[-1]

		# train model 
		if True:
			responses_hat_val = np.zeros((self.num_epochs_total,num_neurons,num_images_val))
			responses_hat_test = np.zeros((self.num_epochs_total,num_neurons,num_images_test))
				
			model = self.initialize_factorized_linear_mapping(num_pixels, num_filters, num_neurons)

			Xtrain = features_train
			Xtest = features_test
			Xval = features_val
			Ytrain = responses_train

			# initialize weights 
			weights = model.get_weights()
			weights[-1] = np.mean(Ytrain,axis=1)
			model.set_weights(weights)

			if self.svd_factorize_flag == True:
				self.initialize_with_svd_Betas(model, Xtrain, Ytrain)

			for iepoch in range(self.num_epochs_total):
				batch_size = 64
				model.fit(Xtrain, Ytrain.T, epochs=10, batch_size=batch_size, shuffle=True, verbose=0)

				responses_hat_val[iepoch] = self.get_model_predicted_responses(model, Xval)
				responses_hat_test[iepoch] = self.get_model_predicted_responses(model, Xtest)

				if verbose:
					print('epoch {:d}'.format(iepoch))

			# choose best epoch with validation data
			R2s_val_over_epochs = np.zeros((self.num_epochs_total,))
			for iepoch in range(self.num_epochs_total):
				R2s_val_over_epochs[iepoch] = np.mean(self.compute_r2_ER(responses_raw_val, responses_hat_val[iepoch]))

			iepoch_max = np.nanargmax(R2s_val_over_epochs)

			# now report R2 test
			R2s_over_neurons = self.compute_r2_ER(responses_raw_test, responses_hat_test[iepoch_max])
			return R2s_over_neurons



	def get_factorized_weights(self, features_train, responses_train, features_test, responses_raw_test):
		#	-- basically, choose some number of epochs and keep the same across models
		#		(you can vary lambdas instead)
		# computes 1 fold of 8-fold cross-validation on features to predict responses
		#		(helpful for hyperparam optimization)
		#	only uses tensorflow mapping
		#   computes performance at each epoch
		#
		# INPUT:
		#	features: (num_images, num_pixels, num_pixels, num_filters), features/embeddings/activations; typically output from a DNN layer
		#	responses_train: (num_neurons, num_images), raw response spike counts
		#	
		# OUTPUT:
		#	model_weights: (list), weights of keras model, where model_weights[ilayer] depends on layer specifics (conv, batchnorm, ...)
		#

		num_images = features_train.shape[0]
		num_neurons = responses_train.shape[0]

		num_pixels = features_train.shape[1]
		num_filters = features_train.shape[-1]

		# train model 
		model = self.initialize_factorized_linear_mapping(num_pixels, num_filters, num_neurons)

		Xtrain = features_train
		Ytrain = responses_train

		# initialize weights 
		weights = model.get_weights()
		weights[-1] = np.mean(Ytrain,axis=1)
		model.set_weights(weights)

		if self.svd_factorize_flag == True:
			self.initialize_with_svd_Betas(model, Xtrain, Ytrain)
				# updates weights in place

		R2_max = 0
		model_weights_max = []

		batch_size = 64
		for iepoch in range(self.num_epochs_total):
			model.fit(Xtrain, Ytrain.T, epochs=10, batch_size=batch_size, shuffle=True, verbose=0)

			R2s = self.evaluate_model(features_test, responses_raw_test, model=model)

			print('epoch {:d}, R2 = {:f}'.format(iepoch, np.median(R2s)))

			if R2_max < np.median(R2s):
				R2_max = np.median(R2s)
				model_weights_max = model.get_weights()

		return model_weights_max


	def initialize_factorized_linear_mapping(self, num_pixels, num_filters, num_neurons, model_weights=None):
		# initializes factorized linear mapping model
		# if given model weights, sets them (useful for distillation responses)

		K.clear_session()

		# set up model
		if True:
			x_input = Input(shape=(num_pixels, num_pixels, num_filters), name='feature_input')
			x = Conv2D(filters=num_neurons, kernel_size=(1,1), kernel_regularizer=l2(self.lambda_mixing), strides=1, padding='same', name='mixing_stage')(x_input)
			x = BatchNormalization()(x)
			x = DepthwiseConv2D(kernel_size=(num_pixels, num_pixels), strides=1, depthwise_regularizer=l2(self.lambda_spatialpool), padding='valid', name='spatial_pool')(x)
			x = Flatten()(x)

			model = Model(inputs=x_input, outputs=x)
			optimizer = Adam(learning_rate=self.learning_rate, clipvalue=0.5)
			model.compile(optimizer=optimizer, loss='mean_squared_error')

			if model_weights is not None:
				model.set_weights(model_weights)

		return model


	def get_model_predicted_responses(self, model, features):
		# DOCUMENT

		num_images = features.shape[0]
		batch_size = 64
		responses_hat = []
		for ibatch in range(0,num_images,batch_size):
			responses_hat.append(model(features[ibatch:ibatch+batch_size]))
		responses_hat = np.concatenate(responses_hat, axis=0).T
			# (num_neurons, num_images)

		return responses_hat


	def evaluate_model(self, features_test, responses_raw_test, model=None, model_weights=None):
		# DOCUMENT

		num_pixels = features_test.shape[1]
		num_filters = features_test.shape[-1]
		num_neurons = responses_raw_test.shape[0]

		if model_weights is not None:
			model = self.initialize_factorized_linear_mapping(num_pixels, num_filters, num_neurons, model_weights)

		responses_hat = self.get_model_predicted_responses(model, features_test)

		R2s = self.compute_r2_ER(responses_raw_test, responses_hat)

		return R2s




	# def initialize_factorized_linear_mapping_old(self, num_pixels, num_filters, num_neurons):
	# 	# initializes factorized linear mapping model

	# 	K.clear_session()

	# 	# set up model
	# 	if True:
	# 		x_input = Input(shape=(num_pixels, num_pixels, num_filters), name='feature_input')
	# 		x = Conv2D(filters=num_neurons, kernel_size=(1,1), kernel_regularizer=l2(self.lambda_mixing), strides=1, padding='same', name='mixing_stage')(x_input)
	# 		x = DepthwiseConv2D(kernel_size=(num_pixels, num_pixels), strides=1, depthwise_regularizer=l2(self.lambda_spatialpool), padding='valid', name='spatial_pool')(x)
	# 		x = Flatten()(x)

	# 		model = Model(inputs=x_input, outputs=x)
	# 		optimizer = Adam(learning_rate=self.learning_rate, clipvalue=0.5)
	# 		model.compile(optimizer=optimizer, loss='mean_squared_error')

	# 	return model


	def initialize_with_svd_Betas(self, model, Xtrain, Ytrain):
		# ridge regression model
		#
		# INPUT:
		#	Xtrain: (num_train_images, num_pixels, num_pixels, num_filters), embedding input for training data
		#	Ytrain: (num_neurons, num_images), repeat-averaged responses
		# OUTPUT:
		#	

		(num_images, num_pixels, _, num_filters) = Xtrain.shape
		(num_neurons, num_images) = Ytrain.shape

		ridger = Ridge(alpha=self.alpha_ridge)

		Xtrain = np.reshape(Xtrain, (num_images,-1))  # (num_images, num_features)

		ridger.fit(Xtrain, Ytrain.T)

		weights = model.get_weights()

		# initialize Betas
		for ineuron in range(num_neurons):
			Betas = ridger.coef_[ineuron,:]
			Betas = np.reshape(Betas, (num_pixels*num_pixels, num_filters))
			U, S, VT = np.linalg.svd(Betas)

			weights[6][:,:,ineuron,0] = np.reshape(U[:,0], (num_pixels,num_pixels)) * np.sqrt(S[0]) # Beta_spatial
			weights[0][0,0,:,ineuron] = VT[0,:] * np.sqrt(S[0])  # Beta_mixing

		model.set_weights(weights)

		return None


	def compute_brain_score(self, responses_true, responses_hat):
		# computes the noise-corrected R2 metric from Bashivan 2019
		#	
		# INPUT:
		#	responses_true: (num_neurons, num_images, num_repeats)
		#	responses_hat: (num_neurons, num_images)
		#
		# OUTPUT:
		#	noise_corrected_R2s: (num_neurons,)
		#
		# code adapted from https://github.com/deanpospisil/er_est/blob/main/er_est.py
		#		function: r2_SB_normed

		num_images = responses_true.shape[1]
		num_neurons = responses_true.shape[0]

		# compute r2(model, mean responses)
		responses_mean = np.nanmean(responses_true,axis=2)
		corrs_model = np.diag(np.corrcoef(responses_mean, responses_hat)[:num_neurons,num_neurons:])

		# compute r2(split1, split2) averaged over 50 runs
		num_runs = 50
		rhos_split = np.zeros((num_runs, num_neurons))
		for irun in range(num_runs):
			responses1, responses2 = self.split_responses(responses_true)
			rhos_split[irun,:] = np.diag(np.corrcoef(responses1, responses2)[:num_neurons,num_neurons:])
		corrs_split = np.mean(rhos_split, axis=0)  # average over runs...not sure why but ok
		corrs_split = 2 * corrs_split / (1 + corrs_split)  # Spearman-brown correction

		# compute ratio
		noise_corrected_R2s = corrs_model**2 / corrs_split  
				## do not square the denominatory --- rho(model vs data) / rho(model vs model) / rho(data vs data)

		return noise_corrected_R2s # (num_neurons,)

		# NOTE: There is huge ambiguity of what Yamins 2014 and Bashivan 2019 actually compute.
		#	I could not find any code from DiCarlo lab to compute their noise-corrected R2 metric.
		#	Instead, I  rely on communication with Dean Pospisil who used their method to compare to his.
		#		He has already been in communication with DiCarlo lab members (e.g., Martin Schrimpf)
		#	Brain-score seems to compute this (not sure if Bashivan uses this though):
				    # r_st = np.corrcoef(x, y.mean(0))[0,1]# get raw correlation
				    # r_tt =  np.corrcoef(y[::2].mean(0), y[1::2].mean(0))[0,1]# get split half correlation
				    # return r_st/r_tt**0.5


	def compute_r2_ER(self, responses_true, responses_hat):
		# computes the noise-corrected R2 metric from Pospisil and Bair, 2021
		#		where ER --> "expected response" provides a consistent R2 estimator (with low bias)
		#	
		# INPUT:
		#	responses_true: (num_neurons, num_images, num_repeats)
		#	responses_hat: (num_neurons, num_images)
		#
		# OUTPUT:
		#	noise_corrected_R2s: (num_neurons,)
		#
		#  code adapted from https://github.com/deanpospisil/er_est/blob/main/er_est.py

		num_images = responses_true.shape[1]
		num_neurons = responses_true.shape[0]
		num_repeats = np.median(np.sum(~np.isnan(responses_true[0,:,:]),axis=1))  # take median number of repeats

		# get mean responses
		responses_mean = np.nanmean(responses_true,axis=2) # (num_neurons,)

		# estimate trial-to-trial variabililty sigma^2
		sigma2 = np.nanmean(np.nanvar(responses_true,axis=2,ddof=1),axis=1) # (num_neurons,)

		# get recentered values
		X_model = responses_hat - np.mean(responses_hat, axis=1)[:,np.newaxis]  # (num_neurons,num_images)
		X_responses = responses_mean - np.mean(responses_mean, axis=1)[:,np.newaxis]

		# compute numerator
		SS_model = np.sum(X_model**2,axis=1)
		SS_model_vs_responses = np.sum(X_model * X_responses, axis=1)**2
				# sum((x-x_mean)(y-y_mean)) all squared
		SS_numerator = SS_model_vs_responses - sigma2/num_repeats * SS_model
				# gets unbiased version, subtracting off correction term

		# compute denominator
		SS_responses = np.sum(X_responses**2,axis=1)
		SS_denominator = SS_model * SS_responses - sigma2/num_repeats * (num_images - 1) * SS_model

		r2_ER = SS_numerator / SS_denominator

		return r2_ER  # (num_neurons,)



	def compute_raw_r2(self, responses_true, responses_hat):
		# computes raw R^2 (fraction variance explained), useful when comparing responses 
		#	for two models (e.g., ensemble and distilled heldout responses)
		#
		# INPUT:
		#	responses_true: (num_neurons, num_images)
		#	responses_hat: (num_neurons, num_images)
		#
		# OUTPUT:
		#	R2s: (num_neurons,) the raw fraction variance explained for each neuron

		# if responses_true/responses_hat only have one neuron, make sure it's a 2d array
		if responses_true.ndim == 1:
			responses_true = responses_true[np.newaxis,:]
		if responses_hat.ndim == 1:
			responses_hat = responses_hat[np.newaxis,:]

		num_neurons = responses_true.shape[0]

		R2s = np.diagonal(np.corrcoef(responses_true, responses_hat)[:num_neurons,num_neurons:])**2

		return np.squeeze(R2s)


	def split_responses(self, responses):
		# splits responses into two groups and returns repeat average for each group
		#
		# INPUT:
		#	responses: (num_neurons, num_images, num_repeats), raw responses
		#	
		# OUTPUT:
		#	responses1: (num_neurons, num_images), repeat-averaged response for group 1
		#	responses2: (num_neurons, num_images), repeat-averaged response for group 2

		num_neurons = responses.shape[0]
		num_images = responses.shape[1]

		responses1 = np.zeros(shape=(num_neurons,num_images))
		responses2 = np.zeros(shape=(num_neurons,num_images))

		mean_rate = np.mean(responses,axis=0)
		for iimage in range(num_images):
			num_repeats = np.sum(~np.isnan(mean_rate[iimage]))
			r = np.random.permutation(num_repeats)

			num_half = int(np.floor(num_repeats/2.))

			responses1[:,iimage] = np.mean(responses[:,iimage,r[:num_half]], axis=-1)
			responses2[:,iimage] = np.mean(responses[:,iimage,r[num_half:2*num_half]], axis=-1)

		return responses1, responses2
