
# Code to plot basic images + responses of any recording session.
# Choose which session_id to examine.
#	
# - plots a sampling of images
# - plots a response heatmap across all images
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
import class_neural_data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


### MAIN SCRIPT

# to run:
#	>> python script1_plot_V4_responses_and_images.py


## choose session
session_ids = [190923,190924,190925,190926,190927,190928,190929,
							201014,201015,201016,201017,201018,201019,201020,201021,201022,201023,201024,201025,
							210224,210225,210226,210301,210302,210303,210304,210305,210308,210309,210310,210312,210315,210316,210322,210323,210324,210325,210326,
							210620, 210621, 211008, 211012, 211013, 211014,211015, 211018, 211022, 211025, 211026, 211027,211028, 211103]

session_id = 211103   # choose a single dataset


## get data
if True:
	I = class_images.ImageClass()
	D = class_neural_data.NeuralDataClass(I)

	imgs, responses = D.get_images_and_responses(session_id, target_size=(112,112))

## plot images
if True:
	num_rows = 10 # feel free to change for more images
	num_cols = 20

	num_images_to_show = num_rows * num_cols

	f = plt.figure(figsize=(num_cols, num_rows))
	num_images = imgs.shape[0]
	img_inds = np.linspace(0,num_images-1, num_images_to_show).astype('int')
	for iimg, img_ind in enumerate(img_inds):
		plt.subplot(num_rows, num_cols, iimg+1)
		plt.imshow(imgs[img_ind].astype('uint8'))
		plt.axis('off')
		plt.title('img {:d}'.format(img_ind))

	f.suptitle('session {:d}'.format(session_id))
	f.tight_layout()
	
	f.savefig('./figures/script1_{:d}_images.pdf'.format(session_id))

## plot response heatmap
if True:
	responses = np.nanmean(responses, axis=-1)
	responses = responses / np.std(responses,axis=1)[:,np.newaxis]  # normalize each neuron

	f = plt.figure(figsize=(10,5))
	v = np.quantile(responses, q=0.95)
	im = plt.imshow(responses, vmin=0, vmax=v, cmap='Reds')
	cbar = plt.colorbar(im, shrink=0.6, aspect=20)
	cbar.set_label('response (normalized)')

	plt.xlabel('image index')
	plt.ylabel('neuron index')
	plt.title('session {:d}'.format(session_id))

	num_neurons = responses.shape[0]
	num_images = responses.shape[1]
	plt.gca().set_aspect(num_images/num_neurons/2)

	f.savefig('./figures/script1_{:d}_responses.pdf'.format(session_id))










