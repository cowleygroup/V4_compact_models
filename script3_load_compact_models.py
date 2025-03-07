
# Code to load a compact model (different versions)
#
# Due to various versions and different packages (keras vs. pytorch), we thought it helpful to show how to load the compact models in various ways and predict responses to images.
# Toggle which section is most useful. For most, either the .keras or .pt (pytorch) files are likely most useful.
#
# Written by Ben Cowley, 2025.
# Tensorflow 2.16.1, keras 3.1.1
#
# Note: This is research code. Feel free to modify this code to work on your system 
#	depending on file structure, versions, etc.



## general imports
import numpy as np 
import os, sys
sys.path.append('./classes')
import class_images
import class_neural_data_test


### session info
if True:
	# choose a random model
	session_id = 201025
	ineuron = 10

### generate images
if True:
	I = class_images.ImageClass()
	D = class_neural_data_test.NeuralDataClass(I)

	imgs, responses = D.get_images_and_responses(isession=0, target_size=(112,112))
	imgs_recentered = I.recenter_images(imgs)  # recenters images for input into compact_models


### load .keras model
if True:
	import class_compact_model

	load_folder = './data_compact_models/models_keras/'
	
	filetag = 'compact_model_{:d}_neuron{:d}'.format(session_id, ineuron)

	M = class_compact_model.CompactModelClass()
	M.load_model(filetag=filetag, load_folder=load_folder)

	responses = M.get_predicted_responses(imgs_recentered) # (num_images,)


### load weights.h5 model
if False:
	import class_compact_model

	folder_nums_filters = './data_compact_models/nums_filters/'
	folder_model_weights_h5 = './data_compact_models/model_weights_h5/'

	filetag = 'compact_model_{:d}_neuron{:d}.weights.h5'.format(session_id, ineuron)

	nums_filters = np.load(folder_nums_filters + 'compact_model_{:d}_neuron{:d}.npy'.format(session_id, ineuron))

	M = class_compact_model.CompactModelClass()
	M.initialize_model(nums_filters=nums_filters)
	M.model.load_weights(folder_model_weights_h5 + filetag)

	responses = M.get_predicted_responses(imgs_recentered) # (num_images,)


### load .h5 model (old keras version)
if False:
	# NOTE: This works for older keras versions (< tf version 2.16.1 and keras < 3.1.1)...this fails for newer versions.
	#		Use .keras model instead.
	import class_compact_model

	folder_model_h5 = './data_compact_models/models_h5/'

	filetag = 'compact_model_{:d}_neuron{:d}'.format(session_id, ineuron)

	M = class_compact_model.CompactModelClass()
	M.initialize_model()
	M.load_model_oldversion(filetag=filetag, load_folder=folder_model_h5)

	responses = M.get_predicted_responses(imgs_recentered) # (num_images,)


### load weights .npy model
if False:
	import class_compact_model

	folder_nums_filters = './data_compact_models/nums_filters/'
	folder_model_weights_npy = './data_compact_models/model_weights_npy/'

	filetag = 'compact_model_{:d}_neuron{:d}.npy'.format(session_id, ineuron)

	nums_filters = np.load(folder_nums_filters + 'compact_model_{:d}_neuron{:d}.npy'.format(session_id, ineuron))

	M = class_compact_model.CompactModelClass()
	M.initialize_model(nums_filters=nums_filters)
	weights = np.load(folder_model_weights_npy + filetag, allow_pickle=True)

	M.model.set_weights(weights)

	responses = M.get_predicted_responses(imgs_recentered) # (num_images,)


### load pytorch model
if False:
	# NOTE: We provide this for pytorch users. However, it is not straightforward to transfer a keras model's weights to a pytorch one.
	#	There are low-level implementation differences between the two packages, and thus we do not expect the exact same
	#	output---but we confirmed that output is very similar between the two packages (< 1% in output response).
	import class_compact_model_pytorch

	load_folder = './data_compact_models/models_torch/'
	nums_filters_folder = './data_compact_models/nums_filters/'
	
	filetag = 'compact_model_{:d}_neuron{:d}'.format(session_id, ineuron)

	M = class_compact_model_pytorch.CompactModelClass()
	M.load_model(filetag=filetag, load_folder=load_folder, nums_filters_folder=nums_filters_folder)

	responses = M.get_predicted_responses(imgs_recentered) # (num_images,)

