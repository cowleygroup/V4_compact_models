# class that extracts all heldout sessions
#
# 4 test sessions (1 session per animal; one animal has 2 (separated by 6 months))
# 1 validation session
# 1 extra held-out session (Bashivan et al., 2019 images)
#
# Use this data for testing prediction performance.
#
# Written by Ben Cowley, 2025.
# Tensorflow 2.16.1, keras 3.1.1
#
# Note: This is research code. Feel free to modify this code to work on your system 
#	depending on file structure, versions, etc.

import numpy as np

class NeuralDataClass:

	def __init__(self, I):
		# INPUT:
		#	I: (ImageClass instance), needs access to class_images
		
		self.data_folder_path = './data_V4_responses/'

		self.I = I  # image class
		self.session_tags = [190923,201025,210225,211022]  # held-out recording sessions for assessing prediction performance
		self.nums_images = [1200, 1200, 1200, 1600]
		self.nums_neurons = [33, 89, 55, 42]

		self.session_tag_hyperparam_testing = 210224  # held-out recording session for hyperparam testing
		self.session_tag_bashivan = 210325 # held-out recording session for comparing with Bashivan et al., 2019 data

		self.num_sessions = 4


	def get_images_and_responses(self, isession, target_size=(224,224)):
		#	returns raw images and responses for given session_tag
		#
		# INPUT:
		#	isession: (integer), indicates which test session (from 0 to 3), 4 sessions total
		#			[0,1,2,3] corresponds to session_ids [190923,201025,210225,211022]
		#	target_size: (2,), tuple of number of pixels for rows and columns e.g., (112,112)
		#
		# OUTPUT:
		#	images: (num_images, num_pixels, num_pixels, 3), raw images (pixel intensities between 0 and 255)
		#	responses: (num_neurons, num_images, num_repeats), neural responses (100ms spike count bins) 
		#				- images may differ in num_repeats; trailing NaNs are placeholders for images with less repeats
		# 				- responses[:,iimage,:] corresponds to images[iimage,:,:,:]

		if isession >= self.num_sessions:
			raise ValueError('isession={:d} is too large. isession must be between 0 and 3, inclusive.'.format(isession))

		# get images
		session_tag = self.session_tags[isession]

		zip_filename = self.data_folder_path + 'images/images_{:d}.zip'.format(session_tag)

		if isession == 3:
			num_images = 1600
		else:
			num_images = 1200

		imgs = self.I.get_images_from_zipfile(zip_filename, np.arange(1,num_images+1), target_size=target_size)
			# note the index, b/c zip files start at 1

		# get split responses
		responses_filename = self.data_folder_path + 'responses_raw/responses_{:d}.npy'.format(session_tag)
		responses = np.load(responses_filename)

		return imgs, responses


	def get_images_and_responses_hyperparam_testing(self, target_size=(224,224)):
		#	returns raw images and responses for hyperparameter search test set (210224)
		#
		# INPUT:
		#	target_size: (2,), tuple of number of pixels for rows and columns e.g., (112,112)
		#
		# OUTPUT:
		#	images: (num_images, num_pixels, num_pixels, 3), raw images (pixel intensities between 0 and 255)
		#	responses: (num_neurons, num_images, num_repeats), neural responses (100ms spike count bins) 
		#				- images may differ in num_repeats; trailing NaNs are placeholders for images with less repeats
		# 				- responses[:,iimage,:] corresponds to images[iimage,:,:,:]

		# get images
		session_tag = self.session_tag_hyperparam_testing

		zip_filename = self.data_folder_path + 'images/images_{:d}.zip'.format(session_tag)
		imgs = self.I.get_images_from_zipfile(zip_filename, np.arange(1,801), target_size=target_size)
			# note the index, b/c zip files start at 1

		# get split responses
		responses_filename = self.data_folder_path + 'responses_raw/responses_{:d}.npy'.format(session_tag)
		responses = np.load(responses_filename)

		return imgs, responses


	def get_images_and_responses_split_for_distilled_responses(self, isession, target_size=(224,224)):
		#  for distillation, we fit a mapping from model embeddings to V4 responses with first half of session's data
		#		and then use the remaining half to evaluate noise-corrected R^2 prediction performance
		# 
		# INPUT:
		#	isession: (integer), indicates which test session (from 0 to 3), 4 sessions total
		#			[0,1,2,3] corresponds to session_ids [190923,201025,210225,211022]
		#	target_size: (2,), tuple of number of pixels for rows and columns e.g., (112,112)
		#
		# OUTPUT:
		#	images_train: (num_train_images, num_pixels, num_pixels, 3), raw images (pixel intensities between 0 and 255)
		#	responses_train: (num_neurons, num_train_images, num_repeats), neural responses (100ms spike count bins); some may be NaNs (for less repeats)
		#	images_train: (num_test_images, num_pixels, num_pixels, 3), raw images (pixel intensities between 0 and 255)
		#	responses_train: (num_neurons, num_test_images, num_repeats), neural responses (100ms spike count bins); some may be NaNs (for less repeats)

		if isession >= self.num_sessions:
			raise ValueError('isession={:d} is too large. isession must be between 0 and 3, inclusive.'.format(isession))

		# get images
		session_tag = self.session_tags[isession]

		zip_filename = self.data_folder_path + 'images/images_{:d}.zip'.format(session_tag)

		if isession == 3:
			num_images = 1600
		else:
			num_images = 1200

		imgs = self.I.get_images_from_zipfile(zip_filename, np.arange(1,num_images+1), target_size=target_size)
			# note the index, b/c zip files start at 1

		# get raw responses
		responses_filename = self.data_folder_path + 'responses_raw/responses_{:d}.npy'.format(session_tag)
		responses = np.load(responses_filename)

		# get training images + responses
		num_images = imgs.shape[0]
		num_half_images = int(num_images/2)
		imgs_train = imgs[:num_half_images]
		responses_train = np.nanmean(responses[:,:num_half_images,:], axis=2)

		# get test images + responses
		imgs_test = imgs[num_half_images:]
		responses_test = responses[:,num_half_images:,:]

		return imgs_train, responses_train, imgs_test, responses_test


	def get_images_and_responses_bashivan_images(self, target_size=(224,224)):
		#	returns raw images and responses for recording session with Bashivan et al., 2019 images (210325)
		#
		# INPUT:
		#	target_size: (2,), tuple of number of pixels for rows and columns e.g., (112,112)
		#
		# OUTPUT:
		#	images: (num_images, num_pixels, num_pixels, 3), raw images (pixel intensities between 0 and 255)
		#	responses: (num_neurons, num_images, num_repeats), neural responses (100ms spike count bins) 
		#				- images may differ in num_repeats; trailing NaNs are placeholders for images with less repeats
		# 				- responses[:,iimage,:] corresponds to images[iimage,:,:,:]

		# get images
		session_tag = self.session_tag_bashivan

		zip_filename = self.data_folder_path + 'images/images_{:d}.zip'.format(session_tag)
		imgs = self.I.get_images_from_zipfile(zip_filename, np.arange(1,641), target_size=target_size)
			# note the index, b/c zip files start at 1

		# get split responses
		responses_filename = self.data_folder_path + 'responses_raw/responses_{:d}.npy'.format(session_tag)
		responses = np.load(responses_filename)

		return imgs, responses





