

# NOTE:
#	This research code works for tensorflow 2.16.1 keras 3.1.1.
#	We provide the code but likely some adaptation will be necessary to work on your system,
#	including retrieving images.
#
#	written by B. Cowley, 2021

import numpy as np 

import os, sys
sys.path.append('../classes')

import class_compact_model
import class_images
import class_ensemble_responses
import class_pruning

import class_neural_data_test
import class_linear_mapping_taskdrivenmodels

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K




### HELPER FUNCTIONS



### MAIN SCRIPT

# to run: python -u script_predict_normal_images.py $igpu $isession > ./output_normalimages_session${isession}.txt

isession = int(sys.argv[2])

nums_neurons = [33, 89, 55, 42]

num_neurons = nums_neurons[isession]

## define classes
if True:

	M = class_compact_model.CompactModelClass()

	I = class_images.ImageClass(ilarge_zip=0)
	R = class_ensemble_responses.EnsembleResponseClass(isession=isession, ilarge_zip=0)

	TS = class_linear_mapping_taskdrivenmodels.LinearMappingClass()


# get test real data (heldout V4 responses)
if True:
	D_test = class_neural_data_test.NeuralDataClass(I)
	imgs_test_real, responses_test_real = D_test.get_images_and_responses(isession=isession, target_size=(112,112))

	imgs_test_real = I.recenter_images(imgs_test_real)

	# only keep heldout responses (1st half was used to train linear mapping)
	num_images = imgs_test_real.shape[0]
	half_num_images = int(np.floor(num_images/2))

	# get test (heldout) data
	imgs_test_real = imgs_test_real[half_num_images:,:,:,:]
	responses_test_real = responses_test_real[:,half_num_images:,:]  # images were chosen randomly to be shown


## compute heldout prediction performance
if True:
	R2s_pruned = np.zeros((num_neurons,))
	R2s_ensemble = np.zeros((num_neurons,))

	load_folder = './train_pruned_models/saved_models_step2/'

	for ineuron in range(num_neurons):

		# R2 pruned predicting real responses
		if True:
			responses_test = responses_test_real[ineuron,:,:][np.newaxis,:,:]

			K.clear_session()
			M.load_model(filetag='pruned_model_step2_session{:d}_neuron{:d}'.format(isession, ineuron), load_folder=load_folder)

			responses_hat = M.get_predicted_responses(imgs_test_real)

			R2s_pruned[ineuron] = TS.compute_unbiased_r2(responses_test, responses_hat[np.newaxis,:])

		# R2 ensemble predicting real responses
		if True:
			responses_hat = R.get_responses_to_realdatatest_session(ineuron=ineuron)

			R2s_ensemble[ineuron] = TS.compute_unbiased_r2(responses_test, responses_hat[np.newaxis,:])

		print('neuron {:d}, R2_ensemble={:f}, R2_pruned={:f}'.format(ineuron, R2s_ensemble[ineuron], R2s_pruned[ineuron]))

	np.save('./results/R2s_ensemble/session{:d}.npy'.format(isession), R2s_ensemble)
	np.save('./results/R2s_step2/session{:d}.npy'.format(isession), R2s_pruned)



