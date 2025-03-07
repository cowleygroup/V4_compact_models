
# NOTE:
#	This research code works for tensorflow 2.16.1 keras 3.1.1.
#	We provide the code but likely some adaptation will be necessary to work on your system,
#	including retrieving images.
#
#	written by B. Cowley, 2021

# given pruned network (without retraining)
# here, retrain for one entire pass
#	and see if performance improves
#	--> reconstruct new network with correct num filters per layer
#	--> then retrain
#	(idea: gradients won't pass through pruned filters anymore, so no need to silence after each block)

import numpy as np 

import os, sys

isession = int(sys.argv[2])
ineuron = int(sys.argv[3])

# if ineuron is out of bounds, quit
if isession == 0 and ineuron >= 33:
	print('ineuron for session {:d} is out of bounds, {:d} neurons total'.format(isession, 33))
	exit(0)
elif isession == 1 and ineuron >= 89:
	print('ineuron for session {:d} is out of bounds, {:d} neurons total'.format(isession, 89))
	exit(0)
elif isession == 2 and ineuron >= 55:
	print('ineuron for session {:d} is out of bounds, {:d} neurons total'.format(isession, 55))
	exit(0)
elif isession == 4 and ineuron >= 42:
	print('ineuron for session {:d} is out of bounds, {:d} neurons total'.format(isession, 42))
	exit(0)	

sys.path.append('../classes')

import class_compact_model
import class_images
import class_ensemble_responses
import class_pruning

import class_neural_data_test
import class_linear_mapping_taskdrivenmodels

import time



### MAIN SCRIPT

# to run:
#	python script_prune_step1_prune_distilled_model.py $igpu $isession $ineuron


isession = int(sys.argv[2])
ineuron = int(sys.argv[3])

## define classes
if True:

	M = class_compact_model.CompactModelClass()

	I = class_images.ImageClass(ilarge_zip=0)
	R = class_ensemble_responses.EnsembleResponseClass(isession=isession, ilarge_zip=0)

	LM = class_linear_mapping_taskdrivenmodels.LinearMappingClass()

	P = class_pruning.PruningClass()


## load step1 pruned model
if True:
	M.load_model(filetag='pruned_model_step1_session{:d}_neuron{:d}'.format(isession, ineuron), load_folder='/jukebox/pillow/bcowley/adroit/data/paper_neural_distill/train_pruned_models/saved_models_step1/')


## get test data (ensemble responses)
if True:
	imgs_test = I.get_test_images_recentered(num_images=5000)
	responses_test = R.get_responses_to_test_images(ineuron=ineuron, num_test_images=5000)


## retrain network
if True:
	for ilarge_zip in range(24):  # train through one pass (24 large zips)

		inds_shuffled = np.random.permutation(500000)

		I = class_images.ImageClass(ilarge_zip=ilarge_zip)
		R = class_ensemble_responses.EnsembleResponseClass(isession=isession, ilarge_zip=ilarge_zip)

		for iblock in range(250): # each block is 2k images
			print('  largezip {:d}, block {:d}'.format(ilarge_zip, iblock))

			inds_train = inds_shuffled[iblock*2000:(iblock+1)*2000]

			P.retrain_model_for_one_block(M, I, R, ineuron, inds_train)

		M.smooth_weights() # smooth weights for interpretability

		responses_hat = M.get_predicted_responses(imgs_test)
		fv = LM.compute_raw_r2(responses_test, responses_hat)[0]

		print('largezip {:d}, fv_ensemble = {:f}'.format(ilarge_zip, fv))

		M.save_model(filetag='pruned_model_step2_session{:d}_neuron{:d}'.format(isession, ineuron), save_folder='./train_pruned_models/saved_models_step2/')







