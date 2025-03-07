
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
import class_images
import class_features

import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



### MAIN SCRIPT

# to run:
#	>> python script2_compute_R2s_taskdriven_vs_ensemblemodel.py $gpu_id


## load classes
if True:
	I = class_images.ImageClass()
	F = class_features.FeaturesClass()

## get test data
if True:
	D_test = class_neural_data_test.NeuralDataClass(I)


## predict task-driven DNNs with ridge regression
if True:
	F.load_model('ResNet50', flattenflag=False)

	LM = class_linear_mapping_ridgereg.LinearMappingClass()

	R2s_taskdriven = []
	for isession in range(D_test.num_sessions):
		imgs_test, responses_test = D_test.get_images_and_responses(isession)

		features_test = F.get_features_from_imgs(imgs_test)

		# long cross-validated computations
		if False:
			alpha = LM.choose_alpha(features_test, responses_test, metric='r2_ER', alpha_min=1e-6, alpha_max=1e5)
			print('alpha = {:f}'.format(alpha))
			R2s = LM.perform_cross_validation(features_test, responses_test, metric='r2_ER', alpha=alpha) # (num_neurons,)
		else: # rough and ready R2s (held out test)
			alpha = 100.
			responses = np.nanmean(responses_test,axis=-1)
			num_images = responses.shape[1]
			inds_train = np.arange(num_images-100)
			inds_val = np.arange(num_images-100,num_images)
			inds_test = inds_val  # test and val indices the same in this case, b/c alpha given

			Y_hat_val, Y_hat_test, _ = LM.get_ridge_regression(features_test[inds_train], responses[:,inds_train], features_test[inds_val], features_test[inds_test], alpha=alpha)
				# ignore Y_hat_val, not used here

			R2s = LM.compute_r2_ER(responses_test[:,inds_test,:], Y_hat_test)

		print('taskdriven DNN, test session {:d}, R2 = {:f}'.format(isession, np.median(R2s)))
		R2s_taskdriven.append(R2s)


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

		imgs_test, responses_test = D_test.get_images_and_responses(isession)

		embeds = np.load(load_folder + 'embeds_test_session{:d}.npy'.format(isession), allow_pickle=True)

		# long cross-validated computations
		if False:
			alpha = LM.choose_alpha(embeds, responses_test, metric='unbiased_r2', alpha_min=1e-6, alpha_max=1e5)
				# note: to speed things up, alpha=100 is a reasonable choice
			print('alpha = {:f}'.format(alpha))

			R2s = LM.perform_cross_validation_for_ensemble(embeds, responses_test, metric='r2_ER', alpha=alpha) # (num_neurons,)
		else: # rough and ready R2s (held out test)
			alpha = 100.
			responses = np.nanmean(responses_test,axis=-1)
			num_images = responses.shape[1]
			inds_train = np.arange(num_images-100)
			inds_val = np.arange(num_images-100,num_images)
			inds_test = inds_val  # test and val indices the same in this case, b/c alpha given
			Y_hat_val, Y_hat_test = LM.get_ensemble_predictions(embeds, responses, inds_train, inds_val, inds_test, alpha=alpha)

			R2s = LM.compute_r2_ER(responses_test[:,inds_test,:], Y_hat_test)

		R2s_over_sessions.append(R2s)

		print('session {:d}, median noise-corrected R2 = {:f}'.format(isession, np.median(R2s)))

	R2s_ensemble = R2s_over_sessions


## plot taskdriven DNN R2s vs ensemble model R2s
if True:
	f = plt.figure()

	plt.plot([0,1],[0,1], '--k')
	for isession in range(D_test.num_sessions):
		plt.plot(R2s_taskdriven[isession], R2s_ensemble[isession], '.', mew=0., ms=10., alpha=0.7)

	plt.xlabel('noise-corrected R2 --- taskdriven DNN')
	plt.ylabel('noise-corrected R2 --- ensemble model')
	
	plt.tight_layout()
	f.savefig('./figures/script2_R2s_taskdriven_vs_ensemblemodel.pdf')









