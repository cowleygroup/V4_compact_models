# class that contains functions relevant to pruning
#
# high-level:
#	step 1: remove bulk of filters that do not contribute to next layer's output
#		remove filters on a layer-by-layer basis (starting with last layer)
#		idea: remove filters that do not help reconstruct output of separable conv layer
#		need to choose how much frac var explained to retain
#		initial sort of filters based on variance of output (i.e., highly-varying filters --> likely contribute a lot)
#		and assess how many of the lowly-varying filters to remove (up to some prediction threshold)
#			question: should we remove means before computing fv? (b/c overall means probably work)  --> not currently done.
#	step 2: retrain pruned models on one full pass (all 12 million images)
#

# NOTES:
#
# A useful dict keeps track of which filters have been removed is the P_struct
#	It keeps track of the layer indices and filter indices for the distilled model for remaining filters.
#	This is helpful for indexing, as each filter can have an overall index (i.e., for all filters)
#		versus a local index for that layer (e.g., between 0 and 99)
#	layer_inds: (num_total_filters,), index of which layer (0 to 4, inclusive)
#	filter_inds: (num_total_filters,), index of filter *for that layer) (0 to 99, inclusive)
#	delta_fvs  (for step3, keeps track of final change in prediction)
#
# There is some trickiness in pruning a candidate filter versus permanently pruning a filter.
#	Thus, care is needed when updating the weights of the model. This is typically done through
#	P_struct, which keeps track of the non-pruned filters. Each time we need to change the weights,
#	we can start with the original weights and then prune any filters that are not in P_struct.
# Each class instance will have its own P, which you can save. Each pruned model has its own P.
#
# Definitions:
#	remaining: filter remains with nonzero weights in model
#	pruned: filter has spatial weights and pooling weights of 0
#	kernel weights: weights of filter in SeparableConv that take a spatial pool of input activations (e.g., the kernel)
#	readout weights: weights of filter in SeparableConv that take a linear combination of outputs from the spatial conv
#	SeparableConv: a spatial convolution followed by readout pooling across the different filters

# Written by B. Cowley, 2025

import numpy as np 

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, Dense, DepthwiseConv2D
from tensorflow.keras.layers import Flatten, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, SGD

from scipy import ndimage

import time
import copy
import pickle


class PruningClass:

	def __init__(self, step1_threshold=0.9, step3_threshold=-0.01):

		self.P = []

		self.step1_threshold = step1_threshold  # fraction of reconstructed frac. var. explained needed to keep 
		self.step3_threshold = step3_threshold # delta frac var explained to keep a filter


	def get_nums_filters(self, M):
		# gets the number of filters per layer based on model in M
		#
		# INPUT:
		#	M: (CompactModelClass instance) distilled or pruned model
		#
		# OUTPUT:
		#	nums_filters: (num_layers,), number of filters per layer

		inds_weights_each_layer = np.array([0,7,14,21,28]) # will tell us how many filters per layer
				# indexes the readout weights
		num_layers = inds_weights_each_layer.size

		weights = M.model.get_weights()

		nums_filters = np.ones((num_layers,)) * np.nan
		for ilayer in range(num_layers):
			nums_filters[ilayer] = weights[inds_weights_each_layer[ilayer]].shape[-1]

		return nums_filters.astype('int')


	def initialize_P_struct(self, M):
		# initializes P to keep track of remaining (non-pruned) filters
		#	P['layer_inds']: (num_remaining_filters,) contains layer inds (0 to 4, inclusive)
		#	P['filter_inds']: (num_remaining_filters,) contains filter inds local to that layer (0 to 99, inclusive)
		#
		# no input or output (change's class's state)

		nums_filters = self.get_nums_filters(M)

		total_num_filters = np.sum(nums_filters)

		P = {}
		P['layer_inds'] = np.zeros((total_num_filters,))
		P['filter_inds'] = np.zeros((total_num_filters,))

		iindex = 0
		for ilayer in range(len(nums_filters)):
			num_filters = nums_filters[ilayer]
			P['layer_inds'][iindex:iindex+num_filters] = ilayer
			P['filter_inds'][iindex:iindex+num_filters] = np.arange(num_filters)
			iindex = iindex + num_filters

		self.P = P


	def remove_filters_from_P(self, inds_filters_to_remove):
		# removes given filters from self.P, which keeps track of remaining filters
		#
		# INPUT:
		#	inds_filters_to_remove: (num_filters,), inds of filters (in absolute reference over all layers + filters)
		#
		# OUTPUT:
		#   none. updates self.P in-place
		#
		# Notes:
		#	- give the user the option of updating self.P. Sometimes you'll need to do that (when you want to keep the
		#		changes) and sometimes not (e.g., just looking at candidate filters to remove)
		#	- you can change the internal P by settig self.P

		# create copy of P
		P_temp = {}

		num_filters_remaining = self.P['layer_inds'].size

		inds_filters_tokeep = np.isin(np.arange(num_filters_remaining), inds_filters_to_remove) == False
		P_temp['layer_inds'] = self.P['layer_inds'][inds_filters_tokeep]
		P_temp['filter_inds'] = self.P['filter_inds'][inds_filters_tokeep]

		self.P = P_temp


	def update_P(self, P):
		# resets self.P to P
		#
		# INPUT:
		#	P: (dict), exact same format as P
		#
		# OUTPUT:
		#	none. updates self.P in-place

		self.P = copy.deepcopy(P)


	def prune_model(self, M):
		# prunes the model of M given self.P
		#	used when you want to update the weights with remaining filters in self.P
		#
		# INPUT:
		#	M: (CompactModelClass), instance of the distilled model
		#
		# OUTPUT:
		#	none. updates model's weights in-place.
		#
		# Notes:
		#	- some trickiness b/c layer 0 is a full conv and the dense beta readout
		#	- intuitive way to think about it is that when we remove filter j from layer i, we remove three things:
		#			- the readout weights for filter j in layer i  (i.e., the column)
		#			- the kernel weights for jth filter in layer i+1
		#			- the readout weights for jth filter in layer i+1 (i.e., the row)
		#		(these are actually in consecutive layers)
		#		this is b/c with SeparableConvs, the kernel weights are linked to the previous layer's filters (b/c depthwise)
		#		so if you want to remove one readout filter, you also need to remove its corresponding kernel weights (in the next layer)
		#		and the corresponding readout weights to that kernel in that next layer

		inds_weights_each_layer = [0,7,14,21,28]  # readout weights (kernel weights are directly before)
												  # note layer 0 is a full conv, so weights[0].shape is (5,5,3,100)
												  # beta weights are at weights[34] (or weights[-2])
		nums_filters = self.get_nums_filters(M)  # you can vary these, but basically fixed for distilled models
		num_layers = len(nums_filters)

		weights = M.model.get_weights()

		for ilayer in range(num_layers):
			num_filters = nums_filters[ilayer]

			if ilayer == num_layers-1:  # last layer's kernel weights are Beta (the dense readout), but we need to reshape Beta to conform
				Beta = np.squeeze(weights[-2])
				Beta = np.reshape(Beta, (28,28,nums_filters[-1]))  # kernel weights for last layer

			for ifilter in range(num_filters):
				if np.any((self.P['layer_inds'] == ilayer) * (self.P['filter_inds'] == ifilter)) == False:  # filter not remaining, so prune
					
					# prune the filter's readout weights
					weights[inds_weights_each_layer[ilayer]][:,:,:,ifilter] = 0.

					# prune the filter's offset weights
					weights[inds_weights_each_layer[ilayer]+1][ifilter] = 0.

					# prune the filter's kernel weights and readout weights (i.e., the row) for the next layer
					if ilayer < num_layers-1:
						weights[inds_weights_each_layer[ilayer+1] - 1][:,:,ifilter,0] = 0.  # minus 1 b/c we are accessing the stage-1 kernel weights
						weights[inds_weights_each_layer[ilayer+1]][0,0,ifilter,:] = 0.  # removes the "row" of readout weights (since this would just be zero anyway)
					else:  # last layer has Beta spatial readouts as its kernel weights
						Beta[:,:,ifilter] = 0.

			if ilayer == num_layers-1:  # reshape pruned Beta back to original shape
				Beta = np.reshape(Beta, (-1,))
				weights[-2][:,0] = Beta

		M.model.set_weights(weights)


	def make_one_layer_models(self, M, ilayer):
		# makes small keras model to replicated one layer
		#	breaks apart the SeparableConv into a model_depth (e.g., the kernel weights)
		#	and model_readout (e.g., the pooling over filters)
		#
		# INPUT:
		#	M: (CompactModelClass instance), distilled model
		#	ilayer: (0 to 3, inclusive), which layer to replicate
		#
		# OUTPUT:
		#	model_depth: (keras model that performs depthwise conv)
		#	model_readout: (keras model that performs dense readout)

		weights = M.model.get_layer('layer{:d}_conv'.format(ilayer+1)).get_weights()

		if ilayer == 0:
			strides_length = 2
			num_input_pixels = 112
			num_output_pixels = 56
		elif ilayer == 1:
			strides_length = 2
			num_input_pixels = 56
			num_output_pixels = 28
		else:
			strides_length = 1
			num_input_pixels = 28
			num_output_pixels = 28

		num_input_filters = weights[0].shape[2]
		num_output_filters = weights[1].shape[-1]

		# replicate depthwise operation
		if True:
			x_input = Input(shape=(num_input_pixels,num_input_pixels,num_input_filters), name='depth_input')
			x = DepthwiseConv2D(kernel_size=(5,5), strides=strides_length, padding='same', name='depth_layer')(x_input)

			model_depth = Model(inputs=x_input, outputs=x)

			weights_depth = model_depth.get_weights() # weights_depth[1] initialized to 0s 
			weights_depth[0] = np.copy(weights[0])
			model_depth.set_weights(weights_depth)

		# replicate dense weight readout
		if True:
			x_input = Input(shape=(num_output_pixels,num_output_pixels,num_input_filters), name='readout_input')
			x = Conv2D(filters=num_output_filters, kernel_size=(1,1), padding='valid', name='readout_layer')(x_input)

			model_readout = Model(inputs=x_input, outputs=x)

			weights_readout = model_readout.get_weights() # weights_readout[1] already initialized to 0s
			weights_readout[0] = np.copy(weights[1])
			model_readout.set_weights(weights_readout)

		return model_depth, model_readout


	def compute_fv_for_layer_output(self, model_readout, output_conv, weights_orig, Y_orig, Y_norm, inds_pruned_filters):
		# quick code to compute the frac. var. explained for the output of a layer (used with prune_layer)
		#
		# INPUT: (bunch of variables)
		# OUTPUT:
		#	fv: (float), frac. var. explained if this filter were pruned

		weights_temp = copy.deepcopy(weights_orig)
		weights_temp[0][0,0,inds_pruned_filters,:] = 0.  # set all readout weights for selected filters to 0
		model_readout.set_weights(weights_temp)

		# compute sum-of-squares (recentering so we don't worry about mean offsets)
		Y_hat = model_readout.predict(output_conv)
		Y_hat_mean = np.mean(np.mean(np.mean(Y_hat,axis=0),axis=0),axis=0)[np.newaxis,np.newaxis,np.newaxis,:]
		Y_orig_mean = np.mean(np.mean(np.mean(Y_orig,axis=0),axis=0),axis=0)[np.newaxis,np.newaxis,np.newaxis,:]
		ss = np.sum(((Y_hat - Y_hat_mean) - (Y_orig - Y_orig_mean))**2)

		fv = 1. - ss / Y_norm  # computes fv (without considering the mean offsets)

		return fv


	def prune_layer(self, M, ilayer, imgs_test):
		# prunes layer's filters based on reconstruction error
		# ** MAKE SURE YOU RUN prune_last_layer FIRST! **
		# 
		# INPUT:
		#	M: (CompactModelClass instance), contains distilled model
		#	ilayer: (integer between 0 to 3 inclusive), layer to remove filters
		#	imgs_test: (num_images,112,112,3), re-centered images as input to distilled model
		#
		# OUTPUT:
		# 	none. updates model and self.P in-place.
		#
		# Notes:
		# 	big picture:
		#	sort each input filter's variance (summed over pixels) across 5k images
		#	then try to reconstruct the output of the conv layer with fewest number of
		#	filters --- prune the filters not needed
		# 	each layer has conv weights (p x p x F_in x 1) and readout weights (1 x 1 x F_in x F_out)
		# 	compute var of input activations convolved with conv weights 
		#	and multiplied by readout weights for each input filter (the targeted filters)
		#	
		# 	- we prune sequentially, starting with the last layer. we could do this in parallel,
		#		but that would try to capture fluctuations of filters in later layers that 
		#		will have been removed
		#	- to get the output after the kernel stage inside the SeparableConv2D, we 
		#		create a one-layer model
		#   - to prune, we consider the filters after the kernel/conv operation. we prune up to K filters
		#		and then ask how well can we reconstruct the output of the readout operation
		#		so we will be pruning the input filters (which seems counterintuitive)

		# get easy-to-work-with models that replicate the depthwise/dense operations of SeparableConv
		model_depth, model_readout = self.make_one_layer_models(M, ilayer)

		# sort filters by variance of output (assuming all other filters are pruned)
		if True:

			weights = model_readout.get_weights()
			W_readout = weights[0][0,0,:,:]  # used to compute s.d.

			num_input_filters = W_readout.shape[0]
			num_output_filters = W_readout.shape[1]

			# get input to layer (output of previous layer)
			activity_model = Model(inputs=M.model.input, outputs=M.model.get_layer('layer{:d}_act'.format(ilayer)).output)
			activation_input = activity_model.predict(imgs_test)

			output_conv = model_depth.predict(activation_input) # (num_images,num_pixels,num_pixels,num_filters)

			# compute variance of each filter (summed over pixels)
			variances = np.ones((num_input_filters,)) * np.nan
			for jfilter in range(num_input_filters):

				# variance over input activations (summed over pixels)
				ss = np.sum((output_conv[:,:,:,jfilter] - np.mean(output_conv[:,:,:,jfilter],axis=0)[np.newaxis,:,:])**2)
				# weighted by readout weights (for that filter only)
				ss = ss * np.sum(W_readout[jfilter,:]**2)  # (works b/c V[aX] = a^2 * V[X])
					# note: ss might be small even if large sum of squares b/c readout weights are small!
				variances[jfilter] = ss

			# now order these variances
			inds_sorted = np.argsort(variances)

		# now identify the bottom K filters that, if pruned, reduce performance > 0.95 * original 
		#	(K is identified with a binary search)
		if True:

			# compute original sum-of-squares with no pruning
			Y_orig = model_readout.predict(output_conv) # (num_images, num_pixels, num_pixels, num_filters)
			Y_orig_mean = np.mean(np.mean(np.mean(Y_orig,axis=0),axis=0),axis=0)[np.newaxis,np.newaxis,np.newaxis,:]
			Y_norm = np.sum((Y_orig - Y_orig_mean)**2)

			weights_orig = model_readout.get_weights()

			ind_filter_threshold = np.floor(num_input_filters/2).astype('int')  # jfilter for which all filters before it have fv > 0.95
			fvs = np.ones((num_input_filters,)) * np.nan

			ind_low = 0
			ind_high = num_input_filters

			# perform binary search to identify ind_filter_threshold
			while True:

				# print([ind_low, ind_filter_threshold, ind_high])

				# compute fv for ind_filter_threshold
				if np.isnan(fvs[ind_filter_threshold]):  # we haven't computed fv for this filter threshold yet
					
					fvs[ind_filter_threshold] = self.compute_fv_for_layer_output(model_readout, output_conv, weights_orig, Y_orig, Y_norm, inds_sorted[:ind_filter_threshold])

				if ind_filter_threshold == 0 and fvs[ind_filter_threshold] < self.step1_threshold:
					break  # removing even the first filter is bad

				if ind_filter_threshold == num_input_filters-1 and fvs[ind_filter_threshold] >= self.step1_threshold:
					break # removing all filters except last one is still good

				# compute fv for ind_filter_threshold - 1
				if np.isnan(fvs[ind_filter_threshold-1]): # fv has not been computed yet

					fvs[ind_filter_threshold-1] = self.compute_fv_for_layer_output(model_readout, output_conv, weights_orig, Y_orig, Y_norm, inds_sorted[:ind_filter_threshold-1])

				# at this point, guaranteed that fvs[i] and fvs[i-1] exist
				#	so now see where we should move ind_filter_threshold

				if fvs[ind_filter_threshold-1] >= self.step1_threshold and fvs[ind_filter_threshold] < self.step1_threshold:
					break  # we've identified the correct threshold! 

				# you haven't found the threshold, so keep searching
				if fvs[ind_filter_threshold] < self.step1_threshold: # threshold will be to the left
					ind_high = ind_filter_threshold - 1
					ind_filter_threshold = np.floor((ind_filter_threshold-ind_low)/2).astype('int') + ind_low
				else:  # threshold will be on the right
					ind_low = ind_filter_threshold + 1
					ind_filter_threshold = np.ceil((ind_high - ind_filter_threshold)/2).astype('int') + ind_filter_threshold

		# the bottom K filters have been identified, so update model and self.P
		if True:

			# make sure you keep at least 3 filters per layer
			ind_filter_threshold = np.clip(ind_filter_threshold, a_min=0, a_max=num_input_filters-3).astype('int')

			inds_filters_to_remove = inds_sorted[:ind_filter_threshold]

			inds_P = np.flatnonzero(self.P['layer_inds'] == ilayer)
			inds_filters = self.P['filter_inds'][inds_P]
			inds_P_to_remove = inds_P[np.isin(inds_filters, inds_filters_to_remove)]
			self.remove_filters_from_P(inds_P_to_remove)  # removes these filters from self.P

			self.prune_model(M)  # updates model by pruning filters in self.P



	def prune_last_layer(self, M, imgs_test, responses_test):
		# step1 pruning for the last layer (which considers the scalar V4 prediction)
		# we need a separate function to:
		#	1) deal with Beta readout
		#	2) consider ensemble responses (not the output of the distilled model) to encourage generalization
		#
		# INPUT:
		#	M: (CompactModelClass), instance of the distilled model
		#	weights_orig: (list with length=num_computational_layers), original weights of the distilled network
		#	imgs_test: (num_images,112,112,3), re-centered images as input to distilled model
		#	responses_test: (num_images,), ensemble responses for the test images
		#
		# OUTPUT:
		#	none. updates M and self.P in-place.
		#
		# Notes:
		# 	Sort filters based on std. dev. of V4 responses. Then identify K, the number of filters
		#		to prune based on the step1_threshold (retaining XX% of the variance)

		# get output from last layer
		if True:		
			ilayer = 4 # last layer
			activity_model = Model(inputs=M.model.input, outputs=M.model.get_layer('layer{:d}_act'.format(ilayer)).output) # output of layer previous to V4 prediction
			output_layer4 = activity_model.predict(imgs_test)  # (num_images,28,28,100)

		# get predicted V4 responses for each filter (to compute s.d.s)
		if True:
			weights = M.model.get_layer('Beta').get_weights()
			w = weights[0]
			w = np.reshape(w, (28,28,-1))
			w = w[np.newaxis,:,:,:]  # (1,28,28,100)

			responses_v4 = np.sum(np.sum(w * output_layer4, axis=1),axis=1)  # (num_images, num_filters)
				# integrates out the spatial readout weights

		# sort filters based on s.d. of output (V4 responses)
		if True:
			num_images = responses_v4.shape[0]
			num_filters = responses_v4.shape[1]
			
			# compute s.d. of each filter
			sds = np.std(responses_v4, axis=0) # (num_filters,)

			inds_sorted = np.argsort(sds)

		# compute original frac var explained (no pruned filters)
		fv_orig = np.corrcoef(np.sum(responses_v4,axis=1), responses_test)[0,1]**2

		# try to prune each sorted filter until we cannot recover threshold performance
		if True:
			inds_filters_to_remove = [] # keeps track of filters to remove (indices for layer 4 only, between 0 and 99 inclusive)

			for ifilter in range(num_filters):
				responses_hat = np.sum(responses_v4[:,inds_sorted[ifilter:]], axis=1)
					# only consider ifilter onwards

				fv = np.corrcoef(responses_hat, responses_test)[0,1]**2

				if fv > self.step1_threshold * fv_orig:
					inds_filters_to_remove.append(inds_sorted[ifilter])
				else:
					break

		# keep at least 3 filters per layer (i.e., take off filters to be removed)
		while num_filters - len(inds_filters_to_remove) < 3:
			inds_filters_to_remove.pop()  # removes last element from list

		inds_P = np.flatnonzero(self.P['layer_inds'] == ilayer)
		inds_filters = self.P['filter_inds'][inds_P]
		inds_P_to_remove = inds_P[np.isin(inds_filters, inds_filters_to_remove)]
		self.remove_filters_from_P(inds_P_to_remove)  # removes these filters from self.P

		self.prune_model(M)  # updates model by pruning filters in self.P


	def sort_P(self, delta_fvs):
		# sorts self.P based on delta_fvs
		#
		# INPUT:
		#	delta_fvs: (num_filters,), delta fv (fv-fv_orig) that marks change in performance
		#			if that filter was pruned
		# no output. updates self.P in-place.
		
		P_temp = {}

		inds_sorted = np.argsort(delta_fvs)
		inds_sorted = inds_sorted[::-1]  # from highest to lowest (increasing importance)

		P_temp['layer_inds'] = self.P['layer_inds'][inds_sorted]
		P_temp['filter_inds'] = self.P['filter_inds'][inds_sorted]

		self.P = P_temp


	def retrain_model_for_one_block(self, M, I, R, ineuron, inds_train):
		# trains pruned model for one block
		#
		# INPUT:
		#	M: (CompactClassModel instance), pruned model
		#	I: (ImageClass instance), image class
		#	R: (EnsembleResponseClass instance), already primed to the correct isession
		#	ineuron: (integer, 0 to N-1), index of neuron from the isession
		#	inds_train: (num_images,), inds to which images to train on
		#
		# OUTPUT:
		#	none. updates M in-place.
		#

		imgs_train = I.get_images_recentered_with_inds(inds_train) # (num_images,num_pixels,num_pixels,3)
		responses_train = R.get_responses(ineuron=ineuron, inds=inds_train) # (num_images,)

		M.train_model(imgs_train, responses_train)  # silences filters after each batch


	def reconstruct_network(self, M):
		# reconstructs a pruned network given M and self.P 
		# M.model is the model that is to be pruned
		#
		# INPUT:
		#	M: (CompactModelClass instance)
		#
		# OUTPUT:
		#	none. updates M in-place (and will overwrite the distilled model)

		# model hyperparameters
		num_layers = 5

		# define architecture
		x_input = Input(shape=(112,112,3), name='image_input')
		x = x_input

		# first layer: 5x5 conv2d (not separable) --- to allow a lot of flexibility
		ilayer = 0
		num_filters = np.sum(self.P['layer_inds'] == ilayer)
		x = Conv2D(filters=num_filters, kernel_size=(5,5), padding='same', name='layer{:d}_conv'.format(ilayer))(x)
		x = BatchNormalization(axis=-1, name='layer{:d}_bn'.format(ilayer))(x)
		x = Activation(activation='relu', name='layer{:d}_act'.format(ilayer))(x)

		for ilayer in range(1, num_layers):
			if ilayer < 3:
				stride_length=2
			else:
				stride_length=1
			num_filters = np.sum(self.P['layer_inds'] == ilayer)
			x = SeparableConv2D(filters=num_filters, kernel_size=(5,5), strides=stride_length, padding='same', name='layer{:d}_conv'.format(ilayer))(x)
			x = BatchNormalization(axis=-1, name='layer{:d}_bn'.format(ilayer))(x)
			x = Activation(activation='relu', name='layer{:d}_act'.format(ilayer))(x)

		# output
		x = Flatten(name='embeddings')(x)
		x = Dense(units=1, name='Beta')(x)   # linear readout layer

		new_model = Model(inputs=x_input, outputs=x)

		optimizer = Adam(learning_rate=1e-4)

		new_model.compile(optimizer=optimizer, loss='mean_squared_error')

		# copy over weights from current model to new model
		if True:

			# layer 0 (full conv)
			ilayer = 0
			inds_filters = self.P['filter_inds'][self.P['layer_inds'] == ilayer].astype('int') # filter inds local to that layer (0 to 99)
			
			# copy over for conv layer 0
			weights = M.model.get_layer('layer{:d}_conv'.format(ilayer)).get_weights()
			weights[0] = weights[0][:,:,:,inds_filters]
			weights[1] = weights[1][inds_filters]
			new_model.get_layer('layer{:d}_conv'.format(ilayer)).set_weights(weights)

			for ilayer in range(1,num_layers):
				# copy over for separable conv layer
				inds_filters_prev = self.P['filter_inds'][self.P['layer_inds'] == ilayer-1].astype('int')
				inds_filters = self.P['filter_inds'][self.P['layer_inds'] == ilayer].astype('int')
				weights = M.model.get_layer('layer{:d}_conv'.format(ilayer)).get_weights()
				weights[0] = weights[0][:,:,inds_filters_prev,:]
				weights[1] = weights[1][:,:,inds_filters_prev,:][:,:,:,inds_filters]
				weights[2] = weights[2][inds_filters]
				new_model.get_layer('layer{:d}_conv'.format(ilayer)).set_weights(weights)

			# copy over for batchnorm layer
			for ilayer in range(num_layers):	
				inds_filters = self.P['filter_inds'][self.P['layer_inds'] == ilayer].astype('int')
				weights = M.model.get_layer('layer{:d}_bn'.format(ilayer)).get_weights()
				for iw in range(len(weights)):
					weights[iw] = weights[iw][inds_filters]
				new_model.get_layer('layer{:d}_bn'.format(ilayer)).set_weights(weights)

			# copy over Beta
			ilayer = num_layers - 1
			inds_filters = self.P['filter_inds'][self.P['layer_inds'] == ilayer].astype('int')
			weights = M.model.get_layer('Beta').get_weights()
			w = weights[0]
			w = np.reshape(w, (28,28,-1))
			w = w[:,:,inds_filters]
			weights[0] = np.reshape(w, (-1,1))
			new_model.get_layer('Beta').set_weights(weights)

		# now update M.model with new model
		del M.model
		M.model = new_model

