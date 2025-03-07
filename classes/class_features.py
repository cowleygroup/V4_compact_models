
# class that accesses internal activity of popular task-driven DNNs available on Keras
#
# - first load the model you are interested in (with various arguments for pooling, etc.)
# - then get features
# - you can also print model summaries as well as get names for relevant layers
#
# Written by Ben Cowley, 2025.
# Tensorflow 2.16.1, keras 3.1.1
#
# Note: This is research code. Feel free to modify this code to work on your system 
#	depending on file structure, versions, etc.

import os
from os import walk
import numpy as np

import tensorflow as tf

import sys
gpu_device = sys.argv[1]
print('using gpu ' + gpu_device)
import os
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_device

from tensorflow.keras import backend as K
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.keras.utils.disable_interactive_logging()

# to see available models for TF 1.14: 
#    https://github.com/tensorflow/docs/tree/r1.14/site/en/api_docs/python/tf/keras/applications

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inceptionv3

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnet_v2

from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet169

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50

from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input as preprocess_input_mobilenet

from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.nasnet import preprocess_input as preprocess_input_nasnetmobile

from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.nasnet import preprocess_input as preprocess_input_nasnetlarge

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception

from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

from tensorflow.keras.models import load_model

from tensorflow.keras import layers
from PIL import Image


class FeaturesClass:

 
### INIT FUNCTION

	def __init__(self):
		# do not load model on onset---b/c we have to call a bunch of classes with different models
		self.model = []
		self.taskdriven_DNN_names = ['VGG19', 'InceptionV3', 'InceptionResNetV2', 'Densenet169', 'ResNet50', 'MobileNet', 'NASNetMobile', 'NASNetLarge', 'Xception']


### CALL FUNCTIONS

	def load_model(self, taskdriven_DNN=None, layer_id='chosen', weights='imagenet', flattenflag=False, avgpoolflag=False, pool_size=(2,2)):
		# loads desired model and layer with trained weights
		#
		# INPUT:
		#	taskdriven_DNN: ('VGG19', 'ResNet50', ...), name of the task-driven DNN
		#	layer_id: ('chosen' or string), name of layer to access internal activity maps; 'chosen' --> chooses pre-specified middle layers
		#								predictive of V4 responses
		#	flattenflag: (True or False), if True, output activity maps are flattened to (num_images, num_features)
		#	avgpoolflag: (True or False), if True, activity maps are average pooled with kernels of pool_size; useful for dim reduction
		#	pool_size: (num_pixels, num_pixels), kernel size/averaging window for the average pool
		#
		# OUTPUT:
		#	None. Loads model internally.
		#
		# NOTE: This calls clear_session, which removes any other DNNs/tensorflow networks on the GPU.

		K.clear_session() # NOTE: Avoids memory leaks; however, this may cause problems by erasing other loaded models!

		self.taskdriven_DNN = taskdriven_DNN

		del self.model

		if self.taskdriven_DNN == 'VGG19':
			self.model = VGG19(weights=weights)
			if layer_id == 'chosen': 
				layer_id = 'block4_pool'
		elif self.taskdriven_DNN == 'InceptionV3':
			self.model = InceptionV3(weights=weights)
			if layer_id == 'chosen':
				layer_id = 'mixed4'
		elif self.taskdriven_DNN == 'InceptionResNetV2':
			self.model = InceptionResNetV2(weights=weights)
			if layer_id == 'chosen': 
				layer_id = 'mixed_6a'
		elif self.taskdriven_DNN == 'Densenet169':
			self.model = DenseNet169(weights=weights)
			if layer_id == 'chosen': 
				layer_id = 'pool3_pool'
		elif self.taskdriven_DNN == 'ResNet50':
			self.model = ResNet50(weights=weights)
			if layer_id == 'chosen': 
				layer_id = 'conv4_block4_out' # old keras: 'activation_33'
		elif self.taskdriven_DNN == 'MobileNet':
			self.model = MobileNet(weights=weights)
			if layer_id == 'chosen': 
				layer_id = 'conv_pw_9_relu'
		elif self.taskdriven_DNN == 'NASNetMobile':
			self.model = NASNetMobile(weights=weights)
			if layer_id == 'chosen': 
				layer_id = 'activation_104'
		elif self.taskdriven_DNN == 'NASNetLarge':
			self.model = NASNetLarge(weights=weights)
			if layer_id == 'chosen': 
				layer_id = 'activation_128'
		elif self.taskdriven_DNN == 'Xception':
			self.model = Xception(weights=weights)
			if layer_id == 'chosen': 
				layer_id = 'add_7'
		else:
			raise NameError('taskdriven_DNN {:s} not recognized'.format(self.taskdriven_DNN))

		x = self.model.get_layer(layer_id).output

		if avgpoolflag == True:
			x = AveragePooling2D(pool_size=pool_size, strides=2, padding='valid')(x)

		if flattenflag == True: # sometimes you don't want it to be flattend 
									 # (e.g., input into ensemble model or for two-stage linear mapping)
			x = Flatten()(x)

		self.model = Model(inputs=self.model.input, outputs=x)


	def get_features_from_imgs(self, imgs_raw):
		# computes DNN features for given (raw) images
		#
		# INPUT:
		#	imgs_raw: (num_images, num_pixels_image, num_pixels_image, 3), images in numpy array with pixel intensities between 0 and 255 (NOT recentered)
		#					- these images will be preprocessed by each DNN's particular preprocessing procedure
		#					- images will be resized as necessary; best shape is 224x224x3.
		# OUTPUT:
		#	features: (num_images, num_pixels_activity, num_pixels_activity, num_filters), DNN features if flattenflag is False.
		#			features[iimage,:,:,ifilter] represents the activity map for the ith filter.
		#			if flattenflag is True, features will have size (num_images, num_pixels_activity^2*num_filters)
		#
		# NOTE: Make sure you call F.load_models() first! Assumes model is loaded.

		imgs = np.copy(imgs_raw)
		
		if self.taskdriven_DNN == 'VGG19':
			imgs = preprocess_input_vgg19(imgs)
		elif self.taskdriven_DNN == 'InceptionV3':
			imgs_processed = []
			for iimg in range(imgs.shape[0]):
				imgs_processed.append(np.array(Image.fromarray(imgs[iimg].astype('uint8')).resize((299,299))))
			imgs_processed = np.array(imgs_processed)
			imgs = preprocess_input_inceptionv3(imgs_processed)
		elif self.taskdriven_DNN == 'InceptionResNetV2':
			imgs_processed = []
			for iimg in range(imgs.shape[0]):
				imgs_processed.append(np.array(Image.fromarray(imgs[iimg].astype('uint8')).resize((299,299))))
			imgs_processed = np.array(imgs_processed)
			imgs = preprocess_input_inception_resnet_v2(imgs_processed)
		elif self.taskdriven_DNN == 'Densenet169':
			imgs = preprocess_input_densenet169(imgs)
		elif self.taskdriven_DNN == 'ResNet50':
			imgs = preprocess_input_resnet50(imgs)
		elif self.taskdriven_DNN == 'MobileNet':
			imgs = preprocess_input_mobilenet(imgs)
		elif self.taskdriven_DNN == 'NASNetMobile':
			imgs = preprocess_input_nasnetmobile(imgs)
		elif self.taskdriven_DNN == 'NASNetLarge':
			imgs_processed = []
			for iimg in range(imgs.shape[0]):
				imgs_processed.append(np.array(Image.fromarray(imgs[iimg].astype('uint8')).resize((331,331))))
			imgs_processed = np.array(imgs_processed)
			imgs = preprocess_input_nasnetlarge(imgs_processed)
		elif self.taskdriven_DNN == 'Xception':
			imgs_processed = []
			for iimg in range(imgs.shape[0]):
				imgs_processed.append(np.array(Image.fromarray(imgs[iimg].astype('uint8')).resize((299,299))))
			imgs_processed = np.array(imgs_processed)
			imgs = preprocess_input_xception(imgs_processed)
		else:
			raise NameError('taskdriven_DNN {:s} not recognized'.format(self.taskdriven_DNN))

		features = self.model.predict(imgs, verbose=False)

		if features.ndim <= 2:
			return features.T # return features as (num_features, num_images) array
		else:
			return features   # return features as (num_images, num_pixels, num_pixels, num_filters) array)


	def get_model_summary(self, taskdriven_DNN=None):
		# prints model summary of chosen taskdriven DNN
		#
		# INPUT:
		#	taskdriven_DNN: ('VGG19', 'InceptionV3', ...), name of taskdriven_DNN
		#
		# OUTPUT:
		#	None. It will output text in the terminal.
		#   useful trick: python -u script.py > output.txt  will output in a text file.
		#
		# NOTE: Calls clear_session(), which removes any other networks/graphs on GPU.

		K.clear_session()  # NOTE: Avoids memory leaks; however, this may cause problems by erasing other loaded models!

		self.taskdriven_DNN = taskdriven_DNN

		del self.model

		if self.taskdriven_DNN == 'VGG19':
			self.model = VGG19(weights='imagenet')
		elif self.taskdriven_DNN == 'InceptionV3':
			self.model = InceptionV3(weights='imagenet')
		elif self.taskdriven_DNN == 'InceptionResNetV2':
			self.model = InceptionResNetV2(weights='imagenet')
		elif self.taskdriven_DNN == 'Densenet169':
			self.model = DenseNet169(weights='imagenet')
		elif self.taskdriven_DNN == 'ResNet50':
			self.model = ResNet50(weights='imagenet')
		elif self.taskdriven_DNN == 'MobileNet':
			self.model = MobileNet(weights='imagenet')
		elif self.taskdriven_DNN == 'NASNetMobile':
			self.model = NASNetMobile(weights='imagenet')
		elif self.taskdriven_DNN == 'NASNetLarge':
			self.model = NASNetLarge(weights='imagenet')
		elif self.taskdriven_DNN == 'Xception':
			self.model = Xception(weights='imagenet')
		else:
			raise NameError('taskdriven_DNN {:s} not recognized'.format(taskdriven_DNN))

		self.model.summary()


	def get_selected_layer_names(self, taskdriven_DNN=None):
		# returns list of useful layer names for specific pretrained DNNs
		#	most are the output of a ReLU stage
		#
		# INPUT:
		#	taskdriven_DNN: ('VGG19', 'InceptionV3', ...), name of taskdriven_DNN
		#
		# OUTPUT:
		#	layer_names: (list of strings), names of relevant/interesting layers of the DNNs
		#			- useful to cycle through for predicting V4 responses
		#			- the most predictive layers are already prespecified in load_model with layer_id='chosen'

		if taskdriven_DNN == None:
			taskdriven_DNN = self.taskdriven_DNN

		if taskdriven_DNN == 'VGG19':
			layer_names = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool']
		elif taskdriven_DNN == 'InceptionV3':
			layer_names = ['mixed0', 'mixed1', 'mixed2', 'mixed3', 'mixed4', 'mixed5', 'mixed6', 'mixed7', 'mixed8', 'mixed9', 'mixed10']
		elif taskdriven_DNN == 'InceptionResNetV2':
			layer_names = ['mixed_5b', 'block35_1_mixed', 'block35_2_mixed', 'block35_3_mixed', 'block35_4_mixed', 'block35_5_mixed', 'block35_6_mixed', 'block35_7_mixed', 'block35_8_mixed', 'block35_9_mixed', 'block35_10_mixed', 'mixed_6a', 'block17_1_mixed', 'block17_2_mixed', 'block17_3_mixed', 'block17_4_mixed', 'block17_5_mixed', 'block17_6_mixed', 'block17_7_mixed', 'block17_8_mixed', 'block17_9_mixed', 'block17_10_mixed', 'block17_11_mixed', 'block17_12_mixed', 'block17_13_mixed', 'block17_14_mixed', 'block17_15_mixed', 'block17_16_mixed', 'block17_17_mixed', 'block17_18_mixed', 'block17_19_mixed', 'block17_20_mixed', 'mixed_7a', 'block8_1_mixed', 'block8_2_mixed', 'block8_3_mixed', 'block8_4_mixed', 'block8_5_mixed', 'block8_6_mixed', 'block8_7_mixed', 'block8_8_mixed', 'block8_9_mixed', 'block8_10_mixed']
		elif taskdriven_DNN == 'Densenet169':
			layer_names = ['pool1', 'pool2_pool', 'pool3_pool', 'pool4_pool']
		elif taskdriven_DNN == 'ResNet50':
			layer_names = ['activation_3', 'activation_6', 'activation_9', 'activation_12', 'activation_15', 'activation_18', 'activation_21', 'activation_24', 'activation_27', 'activation_30', 'activation_33', 'activation_36', 'activation_39', 'activation_42', 'activation_45', 'activation_48']
		elif taskdriven_DNN == 'MobileNet':
			layer_names = ['conv1_relu', 'conv_dw_1_relu', 'conv_pw_1_relu', 'conv_dw_2_relu', 'conv_pw_2_relu', 'conv_dw_3_relu', 'conv_pw_3_relu', 'conv_dw_4_relu', 'conv_pw_4_relu', 'conv_dw_5_relu', 'conv_pw_5_relu', 'conv_dw_6_relu', 'conv_pw_6_relu', 'conv_dw_7_relu', 'conv_pw_7_relu', 'conv_dw_8_relu', 'conv_pw_8_relu', 'conv_dw_9_relu', 'conv_pw_9_relu', 'conv_dw_10_relu', 'conv_pw_10_relu', 'conv_dw_11_relu', 'conv_pw_11_relu', 'conv_dw_12_relu', 'conv_pw_12_relu', 'conv_dw_13_relu', 'conv_pw_13_relu']
		elif taskdriven_DNN == 'NASNetMobile':
			layer_names = ['reduction_add_1_stem_1', 'reduction_add_2_stem_1', 'reduction_add3_stem_1', 'reduction_add4_stem_1', 'reduction_concat_stem_1', 'reduction_bn_1_stem_2', 'reduction_add_1_stem_2', 'reduction_add_2_stem_2', 'reduction_add3_stem_2', 'reduction_concat_stem_2', 'activation_33', 'activation_45', 'activation_57', 'activation_70', 'reduction_add_2_reduce_4', 'reduction_add4_reduce_4', 'activation_82', 'activation_92', 'activation_104', 'activation_116', 'activation_129', 'reduction_concat_reduce_8', 'activation_151', 'activation_163', 'activation_175', 'activation_187']
		elif taskdriven_DNN == 'NASNetLarge':
			layer_names = ['reduction_concat_stem_1', 'activation_22', 'activation_33', 'activation_45', 'activation_57', 'activation_69', 'activation_81', 'activation_94', 'activation_105', 'activation_116', 'activation_128', 'activation_140', 'activation_152', 'activation_164', 'activation_177', 'add_3', 'activation_199', 'activation_211', 'activation_223', 'activation_235', 'activation_247', 'activation_259']
		elif taskdriven_DNN == 'Xception':
			layer_names = ['add', 'add_1', 'add_2', 'add_3', 'add_4', 'add_5', 'add_6', 'add_7', 'add_8', 'add_9', 'add_10', 'add_11']
		else:
			raise NameError('taskdriven_DNN {:s} not recognized'.format(taskdriven_DNN))

		return layer_names


