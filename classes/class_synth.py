
# class that optimizes synthesized images to maximze/minimize model activity
#
# - given compact model, can optimize max/min synthesized images
# - can also optimize images for given filter
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
import class_compact_model

from scipy import ndimage

import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

from tensorflow.keras import backend as K

import time
import copy
import pickle

from PIL import Image 



class SynthClass:

	def __init__(self, M, I):
		# INPUT:
		#	M (class instance): compact model class with loaded model
		#	I (class instance): image class

		self.M = M
		self.I = I


	def optimize_synth_image(self, opt='max', grayscale_flag=False, return_inner_images=False, num_steps=1000, verbose_flag=True):
		# computes the maximizing/minimizing synth image for a compact model's output
		#	starting from a white noise image
		#
		# INPUT:
		#	opt: {'max' or 'min'}, determines to maximize/minimize model output
		#	grayscale: {True or False}, determines if synth image is grayscale (True) or not
		#	return_inner_images: {True or False}, flag of whether to return images throughout the optimization procedure
		#	num_steps: (integer), number of gradient steps
		#	verbose: {True or False}, determines whether to write text about synth procedure (output optimized response at each step)
		#
		# OUTPUT:
		#	synth_image: (112,112,3) rgb array of synthesized image, raw (pixel intensities between 0 and 255)
		#	response_best: (scalar), model's response to synth_image

		# hyperparameters
		if True:
			step_size=10.
			gaussian_filter_sigma=1.
			num_steps_between_smoothing = 50 
				# chosen through empirical testing; change for different model

		# synthesize image
		if True:
			img_synth = self.get_white_noise_image()

			if return_inner_images == True:
				imgs_inner = []
				imgs_inner.append(np.copy(img_synth))

			if opt == 'max':
				response_best = 0.
			else:
				response_best = 10**5

			img_best = np.copy(img_synth)

			for istep in range(num_steps):
				# use GradientTape to access gradient (used to be nicer with Keras)
				img_synth_tf = tf.convert_to_tensor(img_synth)
				with tf.GradientTape() as tape:
					tape.watch(img_synth_tf)
					output = self.M.model(img_synth_tf)
					loss = tf.reduce_mean(output)

				# normalize grad value to prevent blow-ups/adversarial noise
				grad_value = tape.gradient(loss, img_synth_tf)
				grad_value /= (tf.sqrt(tf.reduce_mean(tf.square(grad_value))) + 1e-5)

				# smooth gradient
				grad_value = ndimage.gaussian_filter(grad_value, (0., gaussian_filter_sigma, gaussian_filter_sigma, 0.))

				if opt == 'max':
					img_synth += step_size * grad_value
				elif opt == 'min':
					img_synth -= step_size * grad_value

				# clip image (to ensure it remains between 0 and 255)
				img_synth = self.clip_recentered_image(img_synth)

				# transform synth image to grayscale
				if grayscale_flag == True:
					img_synth = self.I.set_images_to_grayscale(img_synth, recentered=True)
					
				response = np.squeeze(self.M.model(img_synth))

				if verbose_flag == True:
					print('epoch {:d}, response = {:f}'.format(istep, response))

				if opt == 'max' and response > response_best or opt == 'min' and response < response_best:
					response_best = response
					img_best = np.copy(img_synth)

				# periodically smooth image
				if np.mod(istep, num_steps_between_smoothing) == 0 and istep > 0:
					img_synth = ndimage.gaussian_filter(img_synth, (0., gaussian_filter_sigma, gaussian_filter_sigma, 0.))

					if return_inner_images == True:
						imgs_inner.append(np.copy(img_synth))

			img_best = self.I.set_images_to_raw(img_best)

			if return_inner_images == True:
				imgs_inner = np.squeeze(np.array(imgs_inner))
				imgs_inner = self.I.set_images_to_raw(imgs_inner)

				return np.squeeze(img_best), response_best, imgs_inner
			else:
				return np.squeeze(img_best), response_best


	def clip_recentered_image(self, img_synth):
		# clips the new image to be within range of pixel intensity space (between 0 and 255, but needs to be recentered)
		#	necessary b/c gradient could push intensities beyond 0 or 255.
		#
		# INPUT:
		#	img_synth: (1,112,112,3), rgb array, re-centered (pixel intensities between -128 and 128)
		#
		# OUTPUT:
		#	img_synth: (1,112,112,3), rgb array, re-centered; when raw, values are clipped between 0 and 255
		
		rgb_pixel_means = np.array([116.222, 109.270, 100.381])  # computed from our large image dataset, 112x112

		upper_bounds = 255. - rgb_pixel_means
		lower_bounds = 0. - rgb_pixel_means

		for ichannel in range(3):
			img_synth[:,:,:,ichannel] = np.clip(img_synth[:,:,:,ichannel], lower_bounds[ichannel], upper_bounds[ichannel])

		return img_synth


	def get_white_noise_image(self):
		# returns a white noise image
		#
		# INPUT:
		#	None.
		# OUTPUT:
		#	img: (1,112,112,3), white noise image (re-centered)

		img = np.random.uniform(size=(1,112,112,3)) * 50 + 128

		img = self.I.recenter_images(img)

		return img






