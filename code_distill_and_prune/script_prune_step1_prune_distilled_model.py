
# NOTE:
#	This research code works for tensorflow 2.16.1 keras 3.1.1.
#	We provide the code but likely some adaptation will be necessary to work on your system,
#	including retrieving images.
#
#	written by B. Cowley, 2021



# given trained distilled model
# prunes this model:
# 	1. first pass:
# 		- compute delta_fv for each filter
# 		- sort filters based on delta_fv
# 	2. iterative pass:
#		- update delta_fvs for K least important filters
#		- remove filter with highest delta_fv (i.e., least important filter)
# 		- resort delta_fvs
#		- retrain model for 2k images
# 			update delta_fv
# 		- if fv_new - fv_orig < -0.05, stop  (i.e., we see too much of a drop in performance)


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
elif isession ==4 and ineuron >= 42:
	print('ineuron for session {:d} is out of bounds, {:d} neurons total'.format(isession, 42))
	exit(0)

sys.path.append('../classes')

import class_compact_model
import class_images
import class_ensemble_responses
import class_pruning

import time


### MAIN SCRIPT

# to run:
#	python script_prune_step1_prune_distilled_model.py $igpu $isession $ineuron

# define classes
if True:

	M = class_compact_model.CompactModelClass()

	I = class_images.ImageClass(ilarge_zip=0)
	R = class_ensemble_responses.EnsembleResponseClass(isession=isession, ilarge_zip=0)

	P = class_pruning.PruningClass(step1_threshold=0.9)
			# only keep neurons that, together, explain >= 90% of the explained variance of the output of SepConv

## load model
if True:
	load_folder = './train_distilled_models/saved_models/'
	M.load_model(filetag='distilled_model_session{:d}_neuron{:d}'.format(isession, ineuron), load_folder=load_folder)

## get test data
if True:
	imgs_test = I.get_test_images_recentered(num_images=5000)
	responses_test = R.get_responses_to_test_images(ineuron=ineuron, num_test_images=5000)

## heavily-prune model layer-by-layer
if True:

	P.initialize_P_struct(M) # keeps track of which filters have been pruned or not

	# prune each layer (starting from last and working our way towards the first layer)
	print('layer 4...')
	P.prune_last_layer(M, imgs_test, responses_test)

	print('num remaining filters layer 4: {:d}'.format(np.sum(P.P['layer_inds'] == 4).astype('int')))

	for ilayer in [3,2,1,0]:
		print('layer {:d}...'.format(ilayer))
		P.prune_layer(M, ilayer, imgs_test) 

		print('num remaining filters layer {:d}: {:d}'.format(ilayer, np.sum(P.P['layer_inds'] == ilayer).astype('int')))

	# model has been fully pruned, so save it (in preparation for re-training)
	print('reconstructing network...')
	P.reconstruct_network(M)  # resets network to pruned number of filters per layer

	save_folder = './train_pruned_models/saved_models_step1/'
	M.save_model(filetag='pruned_model_step1_session{:d}_neuron{:d}'.format(isession, ineuron), save_folder=save_folder)













