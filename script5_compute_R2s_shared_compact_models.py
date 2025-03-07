

# Code to predict V4 responses to natural images (test sessions).
#	corresponds to Fig. 3d in Cowley et al., 2023
# 
# - uses shared compact models (trained either with teacher models via distillation or directly on V4 data)
# - R2 is computed on the held-out half of images of each test session (the other half was used to estimate the linear mappings between model embeddings and neurons)
# - considers shared compact models:
#		- from ensemble model
#		- linear from ensemble model (e.g., all layers in shared compact model are linear)
#		- from ResNet50
#		- from ResNet50_robust
#		- from CORnet-S
#		- from VGG19
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
import class_linear_mapping_ridgereg
import class_compact_shared_model
import class_images

import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



### MAIN SCRIPT

# to run:
#	>> python script5_compute_R2s_shared_compact_models.py $gpu_id


## load classes
if True:
	I = class_images.ImageClass()
	M = class_compact_shared_model.CompactModelClass()
	LM = class_linear_mapping_ridgereg.LinearMappingClass()

## get test data
if True:
	num_sessions = 4
	session_ids = [190923,201025,210225,211022]
	nums_neurons = [33, 89, 55, 42]
	nums_filters = [5,10,25,35,50,75,100,200]

	D_test = class_neural_data_test.NeuralDataClass(I)

	imgs_test_sessions = []
	responses_test_sessions = []
	for isession, session_id in enumerate(session_ids):
		imgs_train, responses_train, imgs_test, responses_test = D_test.get_images_and_responses_split_for_distilled_responses(isession, target_size=(112,112))
		imgs_test = I.recenter_images(imgs_test)

		imgs_test_sessions.append(imgs_test)
		responses_test_sessions.append(responses_test)


## from ensemble model
if True:
	load_folder = './data_shared_compact_models/from_ensemble_model/'

	R2s_num_filters = []
	for ifilter, num_filters in enumerate(nums_filters):
		R2s_session = []
		ineuron_start = 0 # keeps track of neuron index that starts for each session (shared compact models output 220 neurons)
		for isession, session_id in enumerate(session_ids):
		
			filetag = 'shared_compact_model_{:d}filters'.format(num_filters)
			M.load_model(filetag=filetag, load_folder=load_folder)

			responses_hat = M.get_predicted_responses(imgs_test_sessions[isession])
			responses_hat = responses_hat[ineuron_start:ineuron_start+nums_neurons[isession],:]  # only take this session's neurons

			R2s = LM.compute_r2_ER(responses_test_sessions[isession], responses_hat)
			R2s_session.append(R2s)

			ineuron_start = ineuron_start + nums_neurons[isession]

		R2s_num_filters.append(np.concatenate(R2s_session))

		print('from ensemble model, {:d} filters, R2 = {:f}'.format(num_filters, np.median(R2s_num_filters[-1])))

	R2s_from_ensemble_model = np.array(R2s_num_filters)


## linear from ensemble model
if True:
	load_folder = './data_shared_compact_models/linear_from_ensemble_model/'

	R2s_num_filters = []
	for ifilter, num_filters in enumerate(nums_filters):
		R2s_session = []
		ineuron_start = 0 # keeps track of neuron index that starts for each session (shared compact models output 220 neurons)
		for isession, session_id in enumerate(session_ids):
		
			filetag = 'shared_compact_model_{:d}filters'.format(num_filters)
			M.load_model(filetag=filetag, load_folder=load_folder)

			responses_hat = M.get_predicted_responses(imgs_test_sessions[isession])
			responses_hat = responses_hat[ineuron_start:ineuron_start+nums_neurons[isession],:]  # only take this session's neurons

			R2s = LM.compute_r2_ER(responses_test_sessions[isession], responses_hat)
			R2s_session.append(R2s)

			ineuron_start = ineuron_start + nums_neurons[isession]

		R2s_num_filters.append(np.concatenate(R2s_session))

		print('linear from ensemble model, {:d} filters, R2 = {:f}'.format(num_filters, np.median(R2s_num_filters[-1])))

	R2s_linear_from_ensemble_model = np.array(R2s_num_filters)


## from V4data direct fit
if True:
	load_folder = './data_shared_compact_models/from_V4data_direct_fit/'

	R2s_num_filters = []
	for ifilter, num_filters in enumerate(nums_filters):
		R2s_session = []
		ineuron_start = 0 # keeps track of neuron index that starts for each session (shared compact models output 220 neurons)
		for isession, session_id in enumerate(session_ids):
		
			filetag = 'shared_compact_model_{:d}filters'.format(num_filters)
			M.load_model(filetag=filetag, load_folder=load_folder)

			responses_hat = M.get_predicted_responses(imgs_test_sessions[isession])

			responses_hat = responses_hat[ineuron_start:ineuron_start+nums_neurons[isession],:]  # only take this session's neurons

			R2s = LM.compute_r2_ER(responses_test_sessions[isession], responses_hat)
			R2s_session.append(R2s)

			ineuron_start = ineuron_start + nums_neurons[isession]

		R2s_num_filters.append(np.concatenate(R2s_session))

		print('from V4data direct fit, {:d} filters, R2 = {:f}'.format(num_filters, np.median(R2s_num_filters[-1])))

	R2s_from_V4data_direct_fit = np.array(R2s_num_filters)


## from resnet50
if True:
	load_folder = './data_shared_compact_models/from_resnet50/'

	R2s_num_filters = []
	for ifilter, num_filters in enumerate(nums_filters):
		R2s_session = []
		ineuron_start = 0 # keeps track of neuron index that starts for each session (shared compact models output 220 neurons)
		for isession, session_id in enumerate(session_ids):
		
			filetag = 'shared_compact_model_{:d}filters'.format(num_filters)
			M.load_model(filetag=filetag, load_folder=load_folder)

			responses_hat = M.get_predicted_responses(imgs_test_sessions[isession])
			responses_hat = responses_hat[ineuron_start:ineuron_start+nums_neurons[isession],:]  # only take this session's neurons

			R2s = LM.compute_r2_ER(responses_test_sessions[isession], responses_hat)
			R2s_session.append(R2s)

			ineuron_start = ineuron_start + nums_neurons[isession]

		R2s_num_filters.append(np.concatenate(R2s_session))

		print('from resnet50, {:d} filters, R2 = {:f}'.format(num_filters, np.median(R2s_num_filters[-1])))

	R2s_from_resnet50 = np.array(R2s_num_filters)


## from resnet50 robust
if True:
	load_folder = './data_shared_compact_models/from_resnet50_robust/'

	R2s_num_filters = []
	for ifilter, num_filters in enumerate(nums_filters):
		R2s_session = []
		ineuron_start = 0 # keeps track of neuron index that starts for each session (shared compact models output 220 neurons)
		for isession, session_id in enumerate(session_ids):
		
			filetag = 'shared_compact_model_{:d}filters'.format(num_filters)
			M.load_model(filetag=filetag, load_folder=load_folder)

			responses_hat = M.get_predicted_responses(imgs_test_sessions[isession])
			responses_hat = responses_hat[ineuron_start:ineuron_start+nums_neurons[isession],:]  # only take this session's neurons

			R2s = LM.compute_r2_ER(responses_test_sessions[isession], responses_hat)
			R2s_session.append(R2s)

			ineuron_start = ineuron_start + nums_neurons[isession]

		R2s_num_filters.append(np.concatenate(R2s_session))

		print('from resnet50 robust, {:d} filters, R2 = {:f}'.format(num_filters, np.median(R2s_num_filters[-1])))

	R2s_from_resnet50_robust = np.array(R2s_num_filters)


## from cornets
if True:
	load_folder = './data_shared_compact_models/from_cornets/'

	R2s_num_filters = []
	for ifilter, num_filters in enumerate(nums_filters):
		R2s_session = []
		ineuron_start = 0 # keeps track of neuron index that starts for each session (shared compact models output 220 neurons)
		for isession, session_id in enumerate(session_ids):
		
			filetag = 'shared_compact_model_{:d}filters'.format(num_filters)
			M.load_model(filetag=filetag, load_folder=load_folder)

			responses_hat = M.get_predicted_responses(imgs_test_sessions[isession])
			responses_hat = responses_hat[ineuron_start:ineuron_start+nums_neurons[isession],:]  # only take this session's neurons

			R2s = LM.compute_r2_ER(responses_test_sessions[isession], responses_hat)
			R2s_session.append(R2s)

			ineuron_start = ineuron_start + nums_neurons[isession]

		R2s_num_filters.append(np.concatenate(R2s_session))

		print('from cornets, {:d} filters, R2 = {:f}'.format(num_filters, np.median(R2s_num_filters[-1])))

	R2s_from_cornets = np.array(R2s_num_filters)


## from vgg19
if True:
	load_folder = './data_shared_compact_models/from_vgg19/'

	R2s_num_filters = []
	for ifilter, num_filters in enumerate(nums_filters):
		R2s_session = []
		ineuron_start = 0 # keeps track of neuron index that starts for each session (shared compact models output 220 neurons)
		for isession, session_id in enumerate(session_ids):
		
			filetag = 'shared_compact_model_{:d}filters'.format(num_filters)
			M.load_model(filetag=filetag, load_folder=load_folder)

			responses_hat = M.get_predicted_responses(imgs_test_sessions[isession])
			responses_hat = responses_hat[ineuron_start:ineuron_start+nums_neurons[isession],:]  # only take this session's neurons

			R2s = LM.compute_r2_ER(responses_test_sessions[isession], responses_hat)
			R2s_session.append(R2s)

			ineuron_start = ineuron_start + nums_neurons[isession]

		R2s_num_filters.append(np.concatenate(R2s_session))

		print('from vgg19, {:d} filters, R2 = {:f}'.format(num_filters, np.median(R2s_num_filters[-1])))

	R2s_from_vgg19 = np.array(R2s_num_filters)




## plot ensemble model R2s vs compact model R2s
if True:
	f = plt.figure()

	num_neurons = R2s_from_ensemble_model.shape[1]

	# from_ensemble_model
	R2s_mean = np.mean(R2s_from_ensemble_model, axis=1)
	R2s_stderr = np.std(R2s_from_ensemble_model, axis=1) / np.sqrt(num_neurons)
	plt.errorbar(x=nums_filters, y=R2s_mean, yerr=R2s_stderr, fmt='-', color='xkcd:orange')
	plt.plot(nums_filters, R2s_mean, '.', color='xkcd:orange', label='from_ensemble_model')
	
	# linear_from_ensemble_model
	R2s_mean = np.mean(R2s_linear_from_ensemble_model, axis=1)
	R2s_stderr = np.std(R2s_linear_from_ensemble_model, axis=1) / np.sqrt(num_neurons)
	plt.errorbar(x=nums_filters, y=R2s_mean, yerr=R2s_stderr, fmt='-m')
	plt.plot(nums_filters, R2s_mean, '.m', label='linear_from_ensemble_model')

	# from_V4data_direct_fit
	R2s_mean = np.mean(R2s_from_V4data_direct_fit, axis=1)
	R2s_stderr = np.std(R2s_from_V4data_direct_fit, axis=1) / np.sqrt(num_neurons)
	plt.errorbar(x=nums_filters, y=R2s_mean, yerr=R2s_stderr, fmt='-k')
	plt.plot(nums_filters, R2s_mean, '.k', label='from_V4data_direct_fit')

	# from_resnet50
	R2s_mean = np.mean(R2s_from_resnet50, axis=1)
	R2s_stderr = np.std(R2s_from_resnet50, axis=1) / np.sqrt(num_neurons)
	plt.errorbar(x=nums_filters, y=R2s_mean, yerr=R2s_stderr, fmt='-b')
	plt.plot(nums_filters, R2s_mean, '.b', label='from_resnet50')

	# from_resnet50_robust
	R2s_mean = np.mean(R2s_from_resnet50_robust, axis=1)
	R2s_stderr = np.std(R2s_from_resnet50_robust, axis=1) / np.sqrt(num_neurons)
	plt.errorbar(x=nums_filters, y=R2s_mean, yerr=R2s_stderr, fmt='-r')
	plt.plot(nums_filters, R2s_mean, '.r', label='from_resnet50_robust')

	# from_cornets
	R2s_mean = np.mean(R2s_from_cornets, axis=1)
	R2s_stderr = np.std(R2s_from_cornets, axis=1) / np.sqrt(num_neurons)
	plt.errorbar(x=nums_filters, y=R2s_mean, yerr=R2s_stderr, fmt='-g')
	plt.plot(nums_filters, R2s_mean, '.g', label='from_cornets')

	# from_vgg19
	R2s_mean = np.mean(R2s_from_vgg19, axis=1)
	R2s_stderr = np.std(R2s_from_vgg19, axis=1) / np.sqrt(num_neurons)
	plt.errorbar(x=nums_filters, y=R2s_mean, yerr=R2s_stderr, fmt='-', color='xkcd:brown')
	plt.plot(nums_filters, R2s_mean, '.', color='xkcd:brown', label='from_vgg19')


	plt.xlabel('number of filters in early layers')
	plt.ylabel('noise-corrected R2')
	
	plt.legend()
	plt.tight_layout()
	f.savefig('./figures/script5_R2s_shared_compact_models.pdf')








