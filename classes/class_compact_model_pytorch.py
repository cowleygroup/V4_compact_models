

# compact model class for PyTorch

import numpy as np

import sys
gpu_device = sys.argv[1]
print('using gpu ' + gpu_device)
import os
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_device

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


## implemented separable convolution layer for pytorch (comes automatically in keras)
class SeparableConv2d(nn.Module):
	def __init__(self, num_in_channels, num_out_channels, kernel_size, stride=1, padding='same'):
		# INPUT:
		#   num_in_channels: (int), number of input channels
		#	num_out_channels: (int), number of output channels/filters

		super(SeparableConv2d, self).__init__()

		self.depthwise = nn.Conv2d(num_in_channels, num_in_channels, kernel_size, stride, padding, groups=num_in_channels, bias=False)
		self.pointwise = nn.Conv2d(num_in_channels, num_out_channels, 1, 1, 0, bias=True)  # 1x1 convolution

	def forward(self, x):
		x = self.depthwise(x)
		x = self.pointwise(x)
		return x


class PytorchCompactModel(nn.Module):
	# hard-wired class for implemented 5 layer CNN with varying number of filters; strides + kernel_sizes pre-chosen to match compact models.
	#
	# This class is not meant for training the model in pytorch but retrieving its responses.
	# Future work can include:
	#  - training the model
	#  - accessing model's internal activity (see tensorflow/keras implementation)

	def __init__(self, nums_filters=[100,100,100,100,100]):
		super(PytorchCompactModel, self).__init__()
		self.nums_filters = nums_filters
		self.layers = nn.ModuleDict({
		# first layer: 5x5 conv2d (not separable)
			'layer0_conv': nn.Conv2d(3, nums_filters[0], kernel_size=(5,5), stride=1, padding='same', bias=True),  # Convolutional layer
			'layer0_bn': nn.BatchNorm2d(num_features=nums_filters[0]),
			'layer0_act': nn.ReLU(),
			'layer1_conv_depth': nn.Conv2d(nums_filters[0], nums_filters[0], kernel_size=(5,5), stride=2, padding=0, groups=nums_filters[0], bias=False),
			'layer1_conv_point': nn.Conv2d(nums_filters[0], nums_filters[1], kernel_size=1, stride=1, padding=0, bias=True),
			'layer1_bn': nn.BatchNorm2d(num_features=nums_filters[1]),
			'layer1_act': nn.ReLU(),
			'layer2_conv_depth': nn.Conv2d(nums_filters[1], nums_filters[1], kernel_size=(5,5), stride=2, padding=0, groups=nums_filters[1], bias=False),
			'layer2_conv_point': nn.Conv2d(nums_filters[1], nums_filters[2], kernel_size=1, stride=1, padding=0, bias=True),
			'layer2_bn': nn.BatchNorm2d(num_features=nums_filters[2]),
			'layer2_act': nn.ReLU(),
			'layer3_conv_depth': nn.Conv2d(nums_filters[2], nums_filters[2], kernel_size=5, stride=1, padding='same', groups=nums_filters[2], bias=False),
			'layer3_conv_point': nn.Conv2d(nums_filters[2], nums_filters[3], kernel_size=1, stride=1, padding=0, bias=True),
			'layer3_bn': nn.BatchNorm2d(num_features=nums_filters[3]),
			'layer3_act': nn.ReLU(),
			'layer4_conv_depth': nn.Conv2d(nums_filters[3], nums_filters[3], kernel_size=5, stride=1, padding='same', groups=nums_filters[3], bias=False),
			'layer4_conv_point': nn.Conv2d(nums_filters[3], nums_filters[4], kernel_size=1, stride=1, padding=0, bias=True),
			'layer4_bn': nn.BatchNorm2d(num_features=nums_filters[4]),
			'layer4_act': nn.ReLU(),
			'embeddings': nn.Flatten(start_dim=1),
			'Beta': nn.Linear(nums_filters[4] * 28 * 28,1)
		})

	def forward(self, x):
		x = self.layers['layer0_act'](self.layers['layer0_bn'](self.layers['layer0_conv'](x)))
		x = F.pad(x, (1,3,1,3)) # [left,right,top,bottom] --> need to do this to match keras' 'same' padding
		x = self.layers['layer1_act'](self.layers['layer1_bn'](self.layers['layer1_conv_point'](self.layers['layer1_conv_depth'](x))))
		x = F.pad(x, (1,3,1,3))
		x = self.layers['layer2_act'](self.layers['layer2_bn'](self.layers['layer2_conv_point'](self.layers['layer2_conv_depth'](x))))
		x = self.layers['layer3_act'](self.layers['layer3_bn'](self.layers['layer3_conv_point'](self.layers['layer3_conv_depth'](x))))
		x = self.layers['layer4_act'](self.layers['layer4_bn'](self.layers['layer4_conv_point'](self.layers['layer4_conv_depth'](x))))
		x = x.permute(0,2,3,1)
		x = self.layers['embeddings'](x)
		x = self.layers['Beta'](x)
		return x



class CompactModelClass: 
	# class that defines the "distilled/pruned model"

	def __init__(self):
		self.save_folder = './'

		self.num_layers = 5
		self.batch_size = 64


	def initialize_model(self, nums_filters=[100,100,100,100,100]):
		# initializes compact model
		#
		# INPUT:
		#	num_layers: (integer >= 3), determines number of total layers (includes first rgb conv layer)
		# OUTPUT:
		#	none.

		num_layers = len(nums_filters)

		if num_layers < 3:
			raise ValueError('num_layers needs to be 3 or greater due to striding')

		self.model = PytorchCompactModel(nums_filters)
		
		self.model.cuda()
		self.model.eval()

		
	def get_predicted_responses(self, images_recentered):
		# returns predicted responses
		#
		# INPUT:
		#	images_recentered: (num_images, num_pixels, num_pixels, 3), images already re-centered, 112 x 112
		#
		# OUTPUT:
		#	responses: (num_images,), predicted responses for this compact model

		num_images = images_recentered.shape[0]
		responses = np.zeros((num_images,1))
		for ibatch in range(0,num_images,self.batch_size):
			input_tensor = torch.Tensor(images_recentered[ibatch:ibatch+self.batch_size]).permute(0,3,1,2).to('cuda')
			responses[ibatch:ibatch+self.batch_size] = self.model(input_tensor).detach().to('cpu').numpy()

		return np.squeeze(responses)
		# return np.squeeze(self.model.predict(images_recentered))


	def get_model_weights(self):
		# returns list of weights across layers
		#
		# INPUT:
		#	None.
		# OUTPUT:
		#	weights_pytorch: (list) where weights_pytorch[i] is the weight tensor for the ith layer

		weights_torch = []
		for param in model.parameters():
			weights_torch.append(param.detach().numpy())  # Detach from computation graph and convert to NumPy

		return weights_torch


	def save_model(self, filetag='model', save_folder=None):
		# stores model for later use
		#
		# INPUT:
		#	filetage: (string), file name of the saved model, function will append a '.pt' to it
		#	save_folder: (string), where to save the model. if None, saves to jukebox.
		#			if included, make sure string ends in '/'
		# OUTPUT:
		#	none. saving function.

		if save_folder == None:
			save_folder = self.save_folder

		torch.save(model.state_dict(), save_folder + filetag + '.pt')


	def load_model(self, filetag='model', load_folder=None, nums_filters_folder=None):
		# loads model
		#
		# INPUT:
		#	filetage: (string), file name of the desired model, function will append a '.h5' to it
		#	load_folder: (string), where to load the model. if None, loads from jukebox.
		#			if included, make sure string ends in '/'
		# OUTPUT:
		#	none. loading function.

		if load_folder == None:
			load_folder = self.save_folder

		nums_filters = np.load(nums_filters_folder + filetag + '.npy') # list of 5 ints for the 5 layers
		self.model = PytorchCompactModel(nums_filters)
		self.model.load_state_dict(torch.load(load_folder + filetag + '.pt', weights_only=True))
		self.model.cuda()
		self.model.eval()



