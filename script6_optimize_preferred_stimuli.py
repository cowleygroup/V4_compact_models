

# Code to optimize maximizing synthesized images for a compact model.
# 
# - initial image is white noise
# - this image is modified via gradient ascent on the model's output response
# - over multiple iterations, a preferred stimulus emerges
# - slight smoothing of gradients + images are performed to keep images near natural image manifold
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
import class_synth

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



### MAIN SCRIPT

# to run:
#	>> python script6_optimize_preferred_stimuli.py $gpu_id

session_id = 201025  # four test sessions: [190923,201025,210225,211022]
ineuron = 2

## load model
if True:
	I = class_images.ImageClass()
	M = class_compact_model.CompactModelClass()
	S = class_synth.SynthClass(M,I)

	M.load_model(filetag='compact_model_{:d}_neuron{:d}'.format(session_id, ineuron), load_folder='./data_compact_models/models_keras/')


## generate max synth image
if True:
	f = plt.figure(figsize=(7,7))

	for iimg in range(9):
		print('synth image {:d}'.format(iimg))
		img_synth, response = S.optimize_synth_image(return_inner_images=False, opt='max', num_steps=100)

		plt.subplot(3,3,iimg+1)
		plt.imshow(img_synth.astype('uint8'))
		plt.axis('off')
		plt.title('r = {:.02f}'.format(response))

	f.suptitle('session {:d} neuron {:d}'.format(session_id, ineuron))
	f.tight_layout()
	f.savefig('./figures/script6_{:d}_neuron{:d}_max_synth_images.pdf'.format(session_id, ineuron))



