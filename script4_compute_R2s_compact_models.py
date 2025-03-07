
# Code to predict V4 responses to natural images (test sessions).
#	corresponds to Fig. 1b in Cowley et al., 2023
# 
# - uses ridge regression (final paper uses factorized linear mapping, see classes/class_linear_mapping_taskdrivenmodels.py and classes/class_linear_mapping_ensemble_factorized_sgd.py
#	for factorized mappings)
# - option for cross-validated performance or one fold performance (much faster)
# - considers ResNet50 for task-driven DNN --- others are possible in class_features.py
# - considers ensemble model (25 members) --- DNN weights already trained
# - plots final R2s for ensemble model vs. task-driven DNN
#
# Written by Ben Cowley, 2025.
# Tensorflow 2.16.1, keras 3.1.1
#
# Note: This is research code. Feel free to modify this code to work on your system 
#	depending on file structure, versions, etc.


import numpy as np 

import os, sys
sys.path.append('./classes')

import class_images
import class_neural_data_test
import class_linear_mapping_ensemble_ridgereg
import class_linear_mapping_ridgereg
import class_ensemble_model
import class_compact_model
import class_images
import class_features

import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



### MAIN SCRIPT

# to run:
#	>> python script4_compute_R2s_compact_models.py $gpu_id


## load classes
if True:
	I = class_images.ImageClass()
	F = class_features.FeaturesClass()

## get test data
if True:
	D_test = class_neural_data_test.NeuralDataClass(I)


## get ensemble model embeddings to test sessions and save them
if True:
	M = class_ensemble_model.ModelEnsembleClass(num_members=25)
	load_folder = './data_ensemble_model/'

	F.load_model('ResNet50', flattenflag=False)

	M.load_models(filetag='ensemble_model', load_folder=load_folder)  # loads trained weights

	# save embeds for each session
	for isession in range(D_test.num_sessions):
		embeds = []
		imgs_test, responses_test = D_test.get_images_and_responses(isession)

		features_test = F.get_features_from_imgs(imgs_test)  # get ResNet50 responses as input into ensemble model

		for imember in range(M.num_members):
			embeds.append(M.get_embeddings_for_ith_member(features_test, imember)) # (num_embed_vars,num_images)
				# (num_images, 7,7,512)

			print('session {:d}, ensemble member {:d}'.format(isession, imember))

		np.save('./data_ensemble_model/embeds_test_session{:d}.npy'.format(isession), embeds, allow_pickle=True)
		# Large file!  May need to break up to save for each ensemble member


## predict ensemble model with ridge regression
if True:
	load_folder = './data_ensemble_model/'

	LM = class_linear_mapping_ensemble_ridgereg.LinearMappingClass()

	R2s_over_sessions = []
	for isession in range(D_test.num_sessions):
		print('test session {:d}'.format(isession))

		imgs_train, responses_train, imgs_test, responses_test = D_test.get_images_and_responses_split_for_distilled_responses(isession)

		num_train_images = imgs_train.shape[0]  # first half of images are for training
		num_test_images = imgs_test.shape[0] # second half are for testing

		embeds = np.load(load_folder + 'embeds_test_session{:d}.npy'.format(isession), allow_pickle=True)

		# long cross-validated computations
		if True:
			alpha = 20000.
			inds_train = np.arange(num_train_images)
			inds_val = np.arange(num_train_images, num_train_images + num_test_images)
			inds_test = inds_val  # test and val indices the same in this case, b/c alpha given
			Y_hat_val, Y_hat_test = LM.get_ensemble_predictions(embeds, responses_train, inds_train, inds_val, inds_test, alpha=alpha)

			R2s = LM.compute_r2_ER(responses_test, Y_hat_test)

		R2s_over_sessions.append(R2s)

		print('session {:d}, median noise-corrected R2 = {:f}'.format(isession, np.median(R2s)))

	R2s_ensemble = R2s_over_sessions


## predict compact model prediction performances
if True:
	num_sessions = 4
	session_ids = [190923,201025,210225,211022]
	nums_neurons = [33, 89, 55, 42]

	LM = class_linear_mapping_ridgereg.LinearMappingClass()
	load_folder = './data_compact_models/models_keras/'

	M = class_compact_model.CompactModelClass()

	R2s_compact = []
	for isession, session_id in enumerate(session_ids):

		imgs_train, responses_train, imgs_test, responses_test = D_test.get_images_and_responses_split_for_distilled_responses(isession, target_size=(112,112))

		imgs_test = I.recenter_images(imgs_test)

		R2s = []
		for ineuron in range(nums_neurons[isession]):
			filetag = 'compact_model_{:d}_neuron{:d}'.format(session_id, ineuron)
			M.load_model(filetag=filetag, load_folder=load_folder)

			responses_hat = M.get_predicted_responses(imgs_test) # (num_images,)

			R2 = LM.compute_r2_ER(responses_test[ineuron][np.newaxis,:,:], responses_hat[np.newaxis,:])
			R2s.append(R2)

		R2s_compact.append(np.concatenate(R2s))

		print('session {:d}, median noise-corrected R2 = {:f}'.format(isession, np.median(np.concatenate(R2s))))


## plot ensemble model R2s vs compact model R2s
if True:
	f = plt.figure()

	plt.plot([0,1],[0,1], '--k')
	for isession in range(D_test.num_sessions):
		plt.plot(R2s_compact[isession], R2s_ensemble[isession], '.', mew=0., ms=10., alpha=0.7)

	plt.xlabel('noise-corrected R2 --- compact model')
	plt.ylabel('noise-corrected R2 --- ensemble model')
	
	plt.tight_layout()
	f.savefig('./figures/script4_R2s_compactmodel_vs_ensemblemodel.pdf')

	# NOTE: R2s of the ensemble model are much lower in this plot b/c we used a fast version of 
	#	ridge regression. For the paper, we used a factorized linear mapping with
	#	well estimated hyperparameters.







