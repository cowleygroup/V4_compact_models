# class to access image/response data from any recording session
#
# retrieves images and raw responses
# see script1_plot_V4_responses_and_images.py as an example.
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
		#	I: (ImageClass instance), class_images works with image data

		self.data_folder_path = './data_V4_responses/'

		self.I = I  # image class

		self.session_ids = [190923,190924,190925,190926,190927,190928,190929,
							201014,201015,201016,201017,201018,201019,201020,201021,201022,201023,201024,201025,
							210224,210225,210226,210301,210302,210303,210304,210305,210308,210309,210310,210312,210315,210316,210322,210323,210324,210325,210326,
							210620, 210621, 211008, 211012, 211013, 211014,211015, 211018, 211022, 211025, 211026, 211027,211028, 211103]

		self.nums_images = [1200,1200,1200,1200,1200,1200,1200,300,300,600,
							600,600,1600,1600,1600,2000,2000,2000,1200,800,
							1200,1600,2000,2000,2049,1968,400,2000,2000,2000,
							2000,2000,2000,1200,1200,1200,640,1200,
							1600, 1200, 1200, 2000, 2000, 2000, 2000, 3000,
							1600, 2000, 2000, 2000, 2000, 3000]


	def get_images_and_responses(self, session_id, target_size=(224,224)):
		#	returns raw images and responses for given session_id
		#
		# INPUT:
		#	session_id: (integer), tag to the particular session. e.g., 190924
		#	target_size (optional): (2,), tuple of number of pixels for rows and columns e.g., (112,112)
		# OUTPUT:
		#	images: (num_images, num_pixels, num_pixels, 3), raw images (pixel intensities betweeon 0 and 255)
		#	responses: (num_neurons, num_images, num_repeats), neural responses, taken in 100ms spike count bins

		if session_id not in self.session_ids:
			raise NameError('session tag incorrect: no {:d} found in session tags'.format(session_id))

		isession = self.session_tags.index(session_id)
		num_images = self.nums_images[isession]

		# get images
		zip_filename = self.data_folder_path + 'images/images_{:d}.zip'.format(session_id)
		imgs = self.I.get_images_from_zipfile(zip_filename, np.arange(1,num_images+1), target_size=target_size)
			# note the index, b/c zip files start at 1

		# get responses
		responses_filename = self.data_folder_path + 'responses_raw/responses_{:d}.npy'.format(session_id)
		responses = np.load(responses_filename)

		return imgs, responses


	def compute_SNRs(self, responses_raw):
		# computes SNR of each neuron
		#	Note: SNR is based on SNR_unbiased metric from Pospisil 2021, holds for different number of repeats and images
		#		a threshold of 0.15 is pretty good although some neurons with low spike counts may pass this criterion
		#
		# INPUT: 
		#	responses_raw: (num_neurons, num_images, num_repeats), raw spike count responses over repeats, may contain trailing NaNs
		# OUTPUT:
		#	SNRs: (num_neurons,), SNRs (unbiased) for remaining neurons
		#
		#	Notes:
		#		I could sqrt responses (done in Pospisil work), but I find it doesn't change results much, so not done here.

		num_repeats = np.sum(~np.isnan(responses_raw[0]), axis=1)
		n = np.median(num_repeats)  # number of repeats
		m = responses_raw.shape[1] # number of images

		# compute trial-to-trial var
		vars_per_image = np.nanvar(responses_raw, axis=2, ddof=1)  # ddof=1 is important! should not be 0
		vars_resids = np.mean(vars_per_image, axis=1)  # estimated variance for each neuron

		# compute stimulus var
		vars_stimulus = np.var(np.nanmean(responses_raw, axis=2), axis=1)  # (num_neurons,)
			# the variance of the mean responses

		# unbias d^2 (the biased stimulus var)
		vars_stimulus = vars_stimulus - (m-1)/m * vars_resids / n  # taken from Pospisil 2021

		SNRs = vars_stimulus / vars_resids

		return SNRs

