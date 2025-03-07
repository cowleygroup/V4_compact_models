

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

import class_neural_data_test
import class_linear_mapping_taskdrivenmodels

import time


## MAIN SCRIPT

# python script_train_ensemble_conv_relu.py $gpu_id $isession $ineuron

isession = int(sys.argv[2])  # {0,1,2}, which heldout sessiont to use
ineuron = int(sys.argv[3])	# which neuron from that heldout session to train on


# hyperparameters
if True:
	num_epochs = 1
	num_blocks = 250  # each block ---> 2k images, 250 is one pass through all images

	seed = 31415

	M = class_compact_model.CompactModelClass()

	I = class_images.ImageClass(ilarge_zip=0)
	R = class_ensemble_responses.EnsembleResponseClass(isession=isession, ilarge_zip=0)

	LM = class_linear_mapping_taskdrivenmodels.LinearMappingClass()


# get test data (heldout ensemble responses)
if True:
	imgs_test = I.get_test_images_recentered(num_images=5000)
	responses_test = R.get_responses_to_test_images(ineuron=ineuron, num_test_images=5000)

# get test real data (heldout V4 responses)
if True:
	D_test = class_neural_data_test.NeuralDataClass(I)
	imgs_test_real, responses_test_real = D_test.get_images_and_responses(isession=isession, target_size=(112,112))

	imgs_test_real = I.recenter_images(imgs_test_real)
	responses_test_real = responses_test_real[ineuron,:,:][np.newaxis,:,:]


# train model 
if True:

	M.initialize_model(num_layers=5)  # 100 filters/layer

	for iepoch in range(num_epochs): # one epoch is one pass through all 12M images

		for ilarge_zip in range(24):  # one zip has 500k images in it

			I = class_images.ImageClass(ilarge_zip=ilarge_zip)
			R = class_ensemble_responses.EnsembleResponseClass(isession=isession, ilarge_zip=ilarge_zip)

			# shuffle training indices
			inds_train = np.random.permutation(500000)

			for iblock in range(num_blocks):  # one block is 2k images

				inds_train_block = inds_train[iblock*2000:(iblock+1)*2000]

				imgs_train = I.get_images_recentered_with_inds(inds_train_block)
				responses_train = R.get_responses(ineuron=ineuron, inds=inds_train_block)

				M.train_model(imgs_train, responses_train)  # performs SGD on this block

				if np.mod(iblock,10) == 0:
					print(' pass {:d}, largezip {:d} block {:d}'.format(iepoch, ilarge_zip, iblock))
				
			# smooth weights for better interpretation
			M.smooth_weights()

			# compute R2 between student (distilled) and teacher (ensemble) models
			responses_hat = M.get_predicted_responses(imgs_test)
			fv = LM.compute_raw_r2(responses_test, responses_hat)[0]

			responses_hat = M.get_predicted_responses(imgs_test_real)
			fv_real = LM.compute_unbiased_r2(responses_test_real, responses_hat[np.newaxis,:])[0]

			print(' epoch {:d}, largezip {:d}, fv_ensemble = {:f}, fv_real = {:f}'.format(iepoch, ilarge_zip, fv, fv_real))

			M.save_model(filetag='distilled_model_session{:d}_neuron{:d}'.format(isession, ineuron))

