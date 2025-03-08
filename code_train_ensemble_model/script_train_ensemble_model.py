
# Script to train the ensemble model on all training sessions.
# - hyperparameters were found with exhaustive search (default in ensemble model class)
# - saves performance for each training session on validation set (800 images pepe)
# - performance measured with unbiased CV ridge regression (for final performance, use alternating factorized linear mapping method)
# - epoch/session with largest validation CV performance is saved (early stopping)--> final model
# - saves model for any change in the largest validation performance
#
# 
# Written by Ben Cowley, 2025.
# Tensorflow 2.16.1, keras 3.1.1
#
# Note: This is research code. You will likely need to modify this code to work on your system 
#	depending on file structure, versions, etc.


import numpy as np 
import itertools

import os, sys
sys.path.append('../classes')

import class_neural_data_test
import class_neural_data_train

import class_ensemble_model
import class_images
import class_features
import class_twostage_linear_mapping

from tensorflow.keras import backend as K

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


### MAIN SCRIPT

# to run:
# python -u script_train_ensemble_model.py $GPU_ID > ./output_train.txt

## hyperparameters
if True:
	num_epochs = 5	# number of passes through all training sessions (each session also has K passes for each epoch)
	num_passes_per_set = 1 
	    # number of passes for one session each epoch
	    # telescoping ---> each epoch adds one more pass
		# (needed b/c Beta readout weights are randomized at the start of each training session period)
		# other hyperparameters are defaults in ensemble model class

	learning_rate = 2e0
	LR_decay = 0.85

	save_folder = './'

## load classes
if True:
	seed = 31490

	np.random.seed(seed)

	M = class_ensemble_model.ModelEnsembleClass(num_members=25)

	I = class_images.ImageClass()
	F = class_features.FeaturesClass()

	TS = class_twostage_linear_mapping.LinearMappingClass()
	

## load model and features model
if True:

	F.load_model(pretrained_CNN='ResNet50', flattenflag=False) 
		# will return features with shape (num_images, num_pixels, num_pixels, num_features)

	M.initialize_model(learning_rate=learning_rate)


## get training data
if True:

	D_train = class_neural_data_train.NeuralDataClass(I)


## get test data
if True:
	D_test = class_neural_data_test.NeuralDataClass(I)

	imgs_test, responses_test = D_test.get_images_and_responses_hyperparam_testing()  # heldout validation set
	features_test = F.get_features_from_imgs(imgs_test)


## train ensemble model
if True:

	# keep track of best model (early stopping)
	R2_max = 0.

	M.set_base_model(0)

	for iepoch in range(num_epochs):  # one epoch passes through all sessions once

		# shuffle ordering of sessions
		isessions = np.arange(D_train.num_sessions)
		np.random.shuffle(isessions)

		for jsession in range(D_train.num_sessions):
			isession = isessions[jsession]

			session_tag = D_train.session_tags[isession]
			print('epoch {:d}, training session {:d}'.format(iepoch, session_tag))

			imgs, responses = D_train.get_images_and_responses(isession) # returns raw images and repeat-averaged responses
			features = F.get_features_from_imgs(imgs)

			# z-score responses so that each neuron provides a similarly-sized gradient
			if True:
				m = np.mean(responses,axis=1)[:,np.newaxis]
				s = np.std(responses,axis=1)[:,np.newaxis]
				responses = (responses - m) / s   # z-scoring

			# train model
			for ipass in range(num_passes_per_set):  # one pass through one session --> num_passes_per_set increments by one each time (so more time for each session)
				M.train_models(features, responses, ipass=ipass)

			# track performance for this session
			if True:
				embeds = []
				for imember in range(M.num_members):
					X = M.get_embeddings_for_ith_member(features_test, imember)
					embeds.append(X)

				R2s = TS.perform_cross_validation_for_ensemble(embeds, responses_test, mapping='ridge')

				R2 = np.median(R2s)
				print('cross-validated noise-corrected R2 = {:f}'.format(R2))

			# check if model is the best
			if R2 > R2_max:
				print('saving models...')
				R2_max = R2
				M.save_models(filetag='ensemble_model', save_folder=save_folder)

		# decay learning rate for each epoch
		learning_rate = learning_rate * LR_decay
		M.change_learning_rate(learning_rate)  # updates learning rate for all members

		num_passes_per_set = num_passes_per_set + 1




