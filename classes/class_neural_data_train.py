# class that extracts all training sessions (45 total across 3 animals)
#
# retrieves images and repeat-averaged responses for training ensemble model
#
# Written by Ben Cowley, 2025.
# Tensorflow 2.16.1, keras 3.1.1
#
# Note: This is research code. Feel free to modify this code to work on your system 
#	depending on file structure, versions, etc.

import numpy as np


class NeuralDataClass:

	def get_images_and_responses(self, isession, target_size=(224,224)):
		#	returns raw images and responses for given session_tag
		#
		# INPUT:
		#	isession: (integer), indicates which training session (from 0 to 44), 45 sessions total
		#	target_size (optional): (2,), tuple of number of pixels for rows and columns e.g., (112,112)
		# OUTPUT:
		#	images: (num_images, num_pixels, num_pixels, 3), raw images (pixel intensities between 0 and 255)
		#	responses: (num_neurons, num_images), neural responses (100ms spike count bins), repeat-averaged

		if isession >= self.num_sessions:
			raise ValueError('isession={:d} is too large. isession must be between 0 and 44, inclusive.'.format(isession))

		# get images
		session_tag = self.session_tags[isession]
		inds_imgs = self.inds_imgs[isession]

		zip_filename = self.data_folder_path + 'images/images_{:d}.zip'.format(session_tag)
		imgs = self.I.get_images_from_zipfile(zip_filename, inds_imgs, target_size=target_size)
			# note the index, b/c zip files start at 1

		# get responses
		responses_filename = self.data_folder_path + 'responses_repeat_averaged/responses_{:d}.npy'.format(session_tag)
		responses = np.load(responses_filename)
		responses = responses[:,inds_imgs-1]  # minus one b/c responses are indexed starting at 0

		return imgs, responses


	def __init__(self, I):
		# INPUT:
		#	I: (ImageClass instance), class_images works with image data
		
		self.data_folder_path = './data_V4_responses/'

		self.I = I  # image class
		
		self.session_tags = [190924,190925,190926,190927,190928,190929,  # wilee
							201016,201017,201018,201019,201020,201021,201022,201023,201024,  # pepe
							210226,210301,210302,210303,210304,210305,210308,210309,210310,  # pepe
							210312,210315,210316,210322,210323,210324,210325,210326,  # pepe
							210620,210621,211008,211012,211013,211014,211015,  # rafiki
							211018,211025,211026,211027,211028,211103]  # rafiki


		self.num_sessions = 45

		self.inds_imgs = [np.concatenate([np.arange(1,601), np.arange(901,1201)]), # includes normal, largebank, and normal (avoids synth)
						  np.concatenate([np.arange(1,601), np.arange(901,1201)]),
						  np.concatenate([np.arange(1,601), np.arange(901,1201)]),
						  np.concatenate([np.arange(1,601), np.arange(901,1201)]),
						  np.concatenate([np.arange(1,601), np.arange(901,1201)]),
						  np.concatenate([np.arange(1,601), np.arange(901,1201)]),
						  np.arange(1,601), np.arange(1,601), np.arange(1,601),
						  np.arange(1,1601), np.arange(1,1601), np.arange(1,1601),
						  np.arange(1,2001), np.arange(1,2001), np.arange(1,2001),
						  np.arange(1,1601), np.arange(1,2001), np.arange(1,2001),
						  np.arange(1,2050), np.arange(1,1969), np.arange(1,401),
						  np.arange(1,2001), np.arange(1,2001), np.arange(1,2001),
						  np.arange(1,2001), np.arange(1,2001), np.arange(1,2001),
						  np.arange(1,1201), np.arange(1,1201), np.arange(1,1201),
						  np.arange(1,1201), np.arange(1,1601), np.arange(1,1201),
						  np.arange(1,1201), np.arange(1,2001), np.arange(1,2001), 
						  np.arange(1,2001), np.arange(1,2001), np.arange(1,3001),
						  np.arange(1,2001), np.arange(1,2001), np.arange(1,2001), 
						  np.arange(1,2001), np.arange(1,3001)]




