
# maps embeddings to neural responses
#
# there are a lot of ways to do this:
#	- ridge regression
#	- two-stage mapping, Keras implementation (Klindt 2017)
#	- two-stage mapping, alternating
#	- two-stage mapping, number of factors

import numpy as np
import tensorflow as tf

import sys
import time

from scipy import ndimage
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import ndimage


class LinearMappingClass:


	def __init__(self, alpha=1.):
		self.alpha = alpha


	def choose_alpha(self, features, raw_responses, metric='r2_ER', alpha_min=1e-5, alpha_max=1e5):
		# DOCUMENT
		#   computes 4-fold cross-validation on features to predict responses
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
			responses_hat_val = np.zeros((num_neurons,num_images))
			responses_hat_test = np.zeros((num_neurons,num_images))

			ifold = 1

			inds_test = np.arange(ifold*num_test_images_per_fold, (ifold+1)*num_test_images_per_fold)
			inds_val = np.arange(num_images - (ifold+1)*num_val_images_per_fold, num_images - ifold*num_val_images_per_fold)
			inds_train =  np.array([x for x in range(num_folds * num_test_images_per_fold) if x not in inds_test and x not in inds_val])

			Xtrain = features[inds_train,:,:,:]
			Xtest = features[inds_test,:,:,:]
			Xval = features[inds_val,:,:,:]
			Ytrain = responses_train[:,inds_train]
			num_train_images = Xtrain.shape[0]

			Y_hat_val, Y_hat_test = self.get_ridge_regression(Xtrain, Ytrain, Xval, Xtest, alpha=alpha_min)
			R2_min = np.mean(self.compute_r2_ER(responses[:,inds_val,:], Y_hat_val))

			Y_hat_val, Y_hat_test = self.get_ridge_regression(Xtrain, Ytrain, Xval, Xtest, alpha=alpha_max)
			R2_max = np.mean(self.compute_r2_ER(responses[:,inds_val,:], Y_hat_val))
				
			alpha_middle = alpha_min + (alpha_max - alpha_min) / 2.
			Y_hat_val, Y_hat_test = self.get_ridge_regression(Xtrain, Ytrain, Xval, Xtest, alpha=alpha_middle)
			R2_middle = np.mean(self.compute_r2_ER(responses[:,inds_val,:], Y_hat_val))

			print('initial step')
			print([R2_min, R2_middle, R2_max])

			for istep in range(10):
				if R2_min > R2_middle:
					alpha_max = alpha_middle
					R2_max = R2_middle

					alpha_middle = alpha_min + (alpha_max - alpha_min) / 2.
					Y_hat_val, Y_hat_test = self.get_ridge_regression(Xtrain, Ytrain, Xval, Xtest, alpha=alpha_middle)
					R2_middle = np.mean(self.compute_r2_ER(responses[:,inds_val,:], Y_hat_val))
				elif R2_max > R2_middle:
					alpha_min = alpha_middle
					R2_min = R2_middle

					alpha_middle = alpha_min + (alpha_max - alpha_min) / 2.
					Y_hat_val, Y_hat_test = self.get_ridge_regression(Xtrain, Ytrain, Xval, Xtest, alpha=alpha_middle)
					R2_middle = np.mean(self.compute_r2_ER(responses[:,inds_val,:], Y_hat_val))
				else:  # R2_middle is largest, so choose one side by checking both sides
					alpha_left = alpha_min + (alpha_middle - alpha_min)/2.
					Y_hat_val, Y_hat_test = self.get_ridge_regression(Xtrain, Ytrain, Xval, Xtest, alpha=alpha_left)
					R2_left = np.mean(self.compute_r2_ER(responses[:,inds_val,:], Y_hat_val))
				
					alpha_right = alpha_middle + (alpha_max - alpha_middle)/2.
					Y_hat_val, Y_hat_test = self.get_ridge_regression(Xtrain, Ytrain, Xval, Xtest, alpha=alpha_right)
					R2_right = np.mean(self.compute_r2_ER(responses[:,inds_val,:], Y_hat_val))

					if R2_left > R2_middle and R2_left > R2_right:
						alpha_max=alpha_middle
						R2_max = R2_middle
						alpha_middle = alpha_left
						R2_middle = R2_left
					elif R2_right > R2_middle and R2_right > R2_left:
						alpha_min = alpha_middle
						R2_min = R2_middle
						alpha_middle = alpha_right
						R2_middle = R2_right
					else: # R2_middle greater than both
						alpha_min = alpha_left
						R2_min = R2_left
						alpha_max = alpha_right
						R2_max = R2_right


				print('step {:d}'.format(istep))
				print([R2_min, R2_middle, R2_max])

			if R2_min > R2_middle and R2_min > R2_max:
				return alpha_min
			elif R2_max > R2_middle and R2_max > R2_min:
				return alpha_max
			else:
				return alpha_middle


	def perform_cross_validation(self, features, raw_responses, metric='r2_ER', alpha=None):
		# DOCUMENT
		#   computes 4-fold cross-validation on features to predict responses
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
			responses_hat_val = np.zeros((num_neurons,num_images))
			responses_hat_test = np.zeros((num_neurons,num_images))
			for ifold in range(num_folds):
				print('fold {:d}'.format(ifold))
				inds_test = np.arange(ifold*num_test_images_per_fold, (ifold+1)*num_test_images_per_fold)
				inds_val = np.arange(num_images - (ifold+1)*num_val_images_per_fold, num_images - ifold*num_val_images_per_fold)
				inds_train =  np.array([x for x in range(num_folds * num_test_images_per_fold) if x not in inds_test and x not in inds_val])

				Xtrain = features[inds_train,:,:,:]
				Xtest = features[inds_test,:,:,:]
				Xval = features[inds_val,:,:,:]
				Ytrain = responses_train[:,inds_train]
				num_train_images = Xtrain.shape[0]

				Y_hat_val, Y_hat_test = self.get_ridge_regression(Xtrain, Ytrain, Xval, Xtest, alpha)
			
				responses_hat_val[:,inds_val] = Y_hat_val
				responses_hat_test[:,inds_test] = Y_hat_test

			R2s_biascorrected = self.compute_r2_ER(responses[:,:-4,:], responses_hat_val[:,:-4])
			print('R2_val = {:f}'.format(np.mean(R2s_biascorrected)))

			# now report R2 test
			if metric == 'r2_ER':
				R2s_over_neurons = self.compute_r2_ER(responses[:,:-4,:], responses_hat_test[:,:-4])
				return R2s_over_neurons
			elif metric == 'brainscore':
				R2s_over_neurons = self.compute_r2_ER(responses[:,:-4,:], responses_hat_test[:,:-4])
				return R2s_over_neurons
			elif metric == 'both':
				R2s_brainscore = self.compute_brain_score(responses[:,:-4,:], responses_hat_test[:,:-4])
				R2s_biascorrected = self.compute_r2_ER(responses[:,:-4,:], responses_hat_test[:,:-4])

				return R2s_brainscore, R2s_biascorrected


	def get_ridge_regression(self, Xtrain, Ytrain, Xval, Xtest, alpha=None):
		# ridge regression model
		#
		# INPUT:
		#	Xtrain: (num_train_images, num_pixels, num_pixels, num_filters), embedding input for training data
		#	Ytrain: (num_neurons, num_images), repeat-averaged responses
		#	Xval: (num_val_images, num_pixels, num_pixels, num_filters), embedding input for validation data
		#	Xtest: (num_test_images, num_pixels, num_pixels, num_filters), embedding input for test data
		#	alpha: (optional), sets alpha of ridge regression (default: self.alpha)
		# OUTPUT:
		#	Y_hat_val: (num_neurons, num_val_images), predicted responses for Xval
		#	Y_hat_test: (num_neurons, num_test_images), predicted responses for Xtest

		if alpha is None:
			num_train_images = Xtrain.shape[0]
			Xtilde = np.reshape(Xtrain, (Xtrain.shape[0],-1))
			alpha = 0.01 * np.median(np.var(Xtilde, axis=0)*(num_train_images-1)**2)
			ridger = Ridge(alpha=alpha)
		else:
			ridger = Ridge(alpha=alpha)

		(num_images, num_pixels, _, num_filters) = Xtrain.shape
		(num_neurons, num_images) = Ytrain.shape

		Xtrain = np.reshape(Xtrain, (num_images,-1))  # (num_images, num_features)

		ridger.fit(Xtrain, Ytrain.T)

		Y_hat_val = ridger.predict(np.reshape(Xval, (Xval.shape[0],-1))).T
		Y_hat_test = ridger.predict(np.reshape(Xtest, (Xtest.shape[0],-1))).T

		return Y_hat_val, Y_hat_test, ridger.coef_



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


		
