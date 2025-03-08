
# Script to predict V4 responses in test sessions with ensemble model.
# - uses either ridge regression or factorized linear mapping
# 
# Written by Ben Cowley, 2025.
# Tensorflow 2.16.1, keras 3.1.1
#
# Note: This is research code. You will likely need to modify this code to work on your system 
#	depending on file structure, versions, etc.


import numpy as np 

import os, sys
sys.path.append('../classes')

import class_neural_data_test

import class_ensemble_model
import class_images
import class_features
import class_linear_mapping_ensemble_ridgereg
import class_linear_mapping_ensemble_factorized_sgd




### MAIN SCRIPT


## load classes
if True:
	M = class_ensemble_model.ModelEnsembleClass(num_members=25)

	I = class_images.ImageClass()

	F = class_features.FeaturesClass()
	F.load_model('ResNet50', flattenflag=False)

	D_test = class_neural_data_test.NeuralDataClass(I)


## load model
if True:
	M.load_models(filetag='ensemble_model', load_folder='./')


### predict responses for heldout normal images (4 sessions) with ridge regression
if False:
	LM = class_linear_mapping_ensemble_ridgereg.LinearMappingClass()

	R2s_over_sessions = []
	for isession in range(D_test.num_sessions):

		imgs_test, responses_test = D_test.get_images_and_responses(isession)
		features_test = F.get_features_from_imgs(imgs_test)

		# get embeddings for each ensemble member
		embeddings = []
		for imember in range(M.num_members):
			embeds = M.get_embeddings_for_ith_member(features_test, imember) # (num_embed_vars,num_images)
			embeddings.append(embeds)

		R2s = LM.perform_cross_validation_for_ensemble(embeddings, responses_test, metric='r2_ER', mapping='ridge') # (num_neurons,)

		R2s_over_sessions.append(R2s)

		print('session {:d}, median unbiased R2 = {:f}'.format(isession, np.median(R2s)))

	np.save('./R2s_normalimages_ridge_ensemble_model.npy', R2s_over_sessions)



### predict responses for heldout normal images (4 sessions) with alternating factorized linear mapping
if False:
	LM = class_linear_mapping_ensemble_factorized_sgd.LinearMappingClass()

	R2s_over_sessions = []
	for isession in range(D_test.num_sessions):

		imgs_test, responses_test = D_test.get_images_and_responses(isession)
		features_test = F.get_features_from_imgs(imgs_test)

		# get embeddings for each ensemble member
		embeddings = []
		for imember in range(M.num_members):
			embeds = M.get_embeddings_for_ith_member(features_test, imember) # (num_embed_vars,num_images)
			embeddings.append(embeds)

		R2s = LM.perform_cross_validation_for_ensemble(embeddings, responses_test, metric='r2_ER', mapping='alternating') # (num_neurons,)

		R2s_over_sessions.append(R2s)

		print('session {:d}, median unbiased R2 = {:f}'.format(isession, np.median(R2s)))

	np.save('./R2s_normalimages_alternating_ensemble_model.npy', R2s_over_sessions)






