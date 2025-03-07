
# Image class to access the 12 million image dataset
# 	(as well as images shown in experiments)
#
# Images are stored in 24 "big" zip files --- 500k images per zip file.
#	Images are stored as jpgs in 112x112 resolution with example filename '001234.jpg'
# There's also a heldout zip with 10k images (for testing distilled models to their teacher responses)
#
# Written by Ben Cowley, 2025.
# Tensorflow 2.16.1, keras 3.1.1
#
# Note: This is research code. Feel free to modify this code to work on your system 
#	depending on file structure, versions, etc.


import numpy as np

import zipfile
from PIL import Image
import scipy.ndimage as ndimage

import time

class ImageClass:

	def __init__(self, ilarge_zip=0):
		# re-initialize class if you want to change the zip file index
		# ilarge_zip between 0 and 23, inclusive

		# get image archive, 500k images
		self.image_folder = 'DATA/image_dataset/large_image_zips/'   # CHANGE ME!

		self.archive = zipfile.ZipFile(self.image_folder + 'large_zip{:d}.zip'.format(ilarge_zip), 'r')
			# images stored as '123456.jpg', '001234.jpg' etc.

		self.rgb_mean_pixels = [116.222, 109.270, 100.381]
			# mean pixels for 12 million image dataset
			#	previously identified with script_rgb_mean_pixels.py


	def get_normal_images(self, num_images=1, target_size=(112,112)):
		# returns randomly chosen natural images
		#
		# INPUT:
		#	num_images: (int > 0), number of images to randomly choose
		#	target_size: (2,), tuple of number of pixels for rows and columns e.g., (112,112)
		# OUTPUT:
		#	imgs: (num_images, num_pixels, num_pixels, 3), randomly chosen images (pixel intensities between 0 and 255)

		imgs = np.zeros((num_images, target_size[0], target_size[1], 3))

		for iimg in range(num_images):

			r = np.random.randint(500000)  # zip file has 500k images in it...choose randomly from that
			img_tag = '{:06d}.jpg'.format(r)

			img = Image.open(self.archive.open(img_tag)).resize(size=target_size)
			imgs[iimg] = np.array(img)

		return imgs


	def get_images_with_inds(self, inds_image, target_size=(112,112)):
		# select specific images with indices
		#
		# INPUT:
		#	inds_image: (num_images,), vector of ints between 0 and 500,000 to select specific images
		#	target_size: (2,), tuple of number of pixels for rows and columns e.g., (112,112)
		# OUTPUT:
		#	imgs: (num_images, num_pixels, num_pixels, 3), chosen raw images (pixel intensities from 0 to 255)

		imgs = np.zeros((inds_image.size, target_size[0], target_size[1], 3))

		for iimg in range(inds_image.size):
			ind_image = inds_image[iimg]
			
			img_tag = '{:06d}.jpg'.format(ind_image)

			img = Image.open(self.archive.open(img_tag)).resize(size=target_size)
			imgs[iimg] = np.array(img)

		return imgs


	def get_images_recentered_with_inds(self, inds_image, target_size=(112,112)):
		# for compact model
		#	retrieves specified images in large zip and re-centers (based on dataset means---not ResNet50 processing)
		#		- recentering needed for any compact model.
		#
		# INPUT:
		#	inds_image: (num_images,) integer indices of desired images in large zip
		#			indices need to be in the range [0 and 500000)
		#	target_size: (2,), tuple of number of pixels for rows and columns e.g., (112,112)
		#
		# OUTPUT:
		#	imgs_recentered: (num_images, num_pixels, num_pixels, 3), chosen recentered images (pixel intensities from -128 to 128)

		imgs = np.zeros((inds_image.size, target_size[0], target_size[1], 3))

		for iimg in range(inds_image.size):
			ind_image = inds_image[iimg]
			
			img_tag = '{:06d}.jpg'.format(ind_image)

			img = Image.open(self.archive.open(img_tag)).resize(size=target_size)
			imgs[iimg] = np.array(img)

		for ichannel in range(3):
			imgs[:,:,:,ichannel] = imgs[:,:,:,ichannel] - self.rgb_mean_pixels[ichannel]

		return imgs


	def get_test_images_recentered(self, num_images=10000, ind_img_start=-1, ind_img_end=-1, target_size=(112,112)):
		# retrieves held-out images not in the 12 million
		#	(useful for testing distilled models)
		# Images are recentered for compact models...use set_images_to_raw() to revert back to raw images.
		#
		# INPUT:
		#	num_images: (int from 0 to 10000), number of test images to retrieve (takes the first num_images unless ind_image_start/end are used)
		#	ind_img_start: (integer from 0 to 10000), starting index of image
		# 	ind_img_end: (integer from 0 to 10000), ending index of image + 1, allows user to access range of images
		#		defaults of -1 accesses the first num_images images
		#		else num_images = ind_img_end - ind_img_start
		#	target_size: (2,), tuple of number of pixels for rows and columns e.g., (112,112)
		#
		# OUTPUT:
		#	imgs_recentered: (num_images, num_pixels, num_pixels, 3), recentered images (pixel intensities from -128 to 128)

		image_path = self.image_folder + 'test_10k_images.zip'

		archive_test = zipfile.ZipFile(image_path, 'r')

		if ind_img_start == -1 or ind_img_end == -1:
			ind_img_start = 0
			ind_img_end = num_images
		else:
			num_images = ind_img_end - ind_img_start

		imgs = np.zeros((num_images, target_size[0], target_size[1], 3))
		inds = np.arange(ind_img_start, ind_img_end).astype('int')

		for jimg in range(num_images):
			iimg = inds[jimg]
			img_tag = '{:06d}.jpg'.format(iimg)

			img = Image.open(archive_test.open(img_tag)).resize(size=target_size)  #, resampling=0) nearest neighbor
			imgs[jimg] = np.array(img)

		# recenter images
		for ichannel in range(3):
			imgs[:,:,:,ichannel] = imgs[:,:,:,ichannel] - self.rgb_mean_pixels[ichannel]

		archive_test.close()
		
		return imgs


	def recenter_images(self, imgs_raw):
		# recenters raw images based on means computed across 12M images
		#	- needed for inputting an image into a compact model
		#
		# INPUT:
		#	imgs_raw: (num_images, num_pixels, num_pixels, 3), images (pixel intensities between 0 and 255)
		#
		# OUTPUT:
		#	imgs_recentered: (num_images, num_pixels, num_pixels, 3), recentered images (pixel intensities from -128 to 128)

		imgs = np.copy(imgs_raw)

		if imgs.dtype == 'uint8':
			imgs = imgs.astype('float')

		for ichannel in range(3):
			imgs[:,:,:,ichannel] -= self.rgb_mean_pixels[ichannel]

		return imgs


	def set_images_to_raw(self, imgs_recentered):
		# takes re-centered images (for compact model) and transforms them to raw by adding back the RGB means
		#
		# INPUT:
		#	imgs_recentered: (num_images,num_pixels,num_pixels,3), re-centered images (pixel intensities from -128 to 128)
		#
		# OUTPUT:
		#	imgs_raw: (num_images,num_pixels,num_pixels,3), raw images with pixel intensities between 0 and 255

		imgs = np.copy(imgs_recentered)

		for ichannel in range(3):
			imgs[:,:,:,ichannel] += self.rgb_mean_pixels[ichannel]

		imgs = np.clip(imgs, a_min=0, a_max=255)

		return imgs


	def get_images_from_zipfile(self, zipfileandpath, inds_image, target_size=(112,112)):
		# retrieve images from any given zipfile
		#	- useful for retrieving images shown in experiments
		#	- make sure you check the starting index! shown images started with 1, not 0 (thanks Matlab)
		#
		# INPUT:
		#	zipfileandpath: (string), location of zipfile, e.g., '/DATA/data_V4_responses/images/images_190923.zip'
		#	inds_image: (num_images,), indices to images (assuming images are saved as 'image0001.jpg', etc.)
		#	target_size: (2,), tuple of number of pixels for rows and columns e.g., (112,112)
		#
		# OUTPUT:
		#	imgs_raw: (num_images,num_pixels,num_pixels,3), desired raw images with pixel intensities between 0 and 255

		imgs = []
		archive = zipfile.ZipFile(zipfileandpath, 'r')
		names = archive.namelist()

		for ind_image in inds_image:
			img_tag = 'image{:04d}.jpg'.format(ind_image)

			for name in names: # could be a top folder written in the filenames, so this handles that
				if img_tag in name:
					if target_size[0] != 112 or target_size[1] != 112:
						img = Image.open(archive.open(name)).resize(size=target_size)
					else:
						img = Image.open(archive.open(name))
					img = np.array(img)

					imgs.append(img)
					break

		return np.asarray(imgs)


	def set_images_to_grayscale(self, imgs, recentered_flag=False):
		# given color images, returns grayscale equivalents
		#	 - if already recentered, will convert to grayscale (in raw space) and transform back to recentered
		#
		# INPUT:
		#	imgs: (num_images,num_pixels,num_pixels,3), images (either raw or recentered)
		#	recentered_flag: (True or False), if True, imgs are transformed from recentered to raw (assumes imgs are recentered)
		#
		# OUTPUT:
		#	imgs_grayscale: (num_images,num_pixels,num_pixels,3), grayscale images (raw), pixel intensities between 0 and 255

		if recentered == True:
			imgs_grayscale = self.set_images_to_raw(imgs)
		else:
			imgs_grayscale = np.copy(imgs)

		num_imgs = imgs.shape[0]
		for iimg in range(num_imgs):
			imgs_grayscale[iimg,:,:,:] = np.mean(imgs_grayscale[iimg],axis=-1)[:,:,np.newaxis]

		if recentered == True:
			imgs_grayscale = self.recenter_images(imgs_grayscale)

		return imgs_grayscale


	def get_white_noise_image(self, num_images=1, range_noise=[0,255], target_size=(112,112)):
		# generate white noise images (uniform distribution)
		#
		# INPUT:
		#	num_images: (int), number of white noise images to generate
		#	range_noise: [low_value, high_value], range for the min/max values of the uniform noise
		#	target_size: (2,), tuple of number of pixels for rows and columns e.g., (112,112)
		#
		# OUTPUT:
		#	imgs_whitenoise: (num_images,num_pixels,num_pixels,3), grayscale whitenoise images (raw), pixel intensities between 0 and 255

		imgs = np.random.uniform(low=range_noise[0],high=range_noise[1], size=(num_images, target_size[0],target_size[1],1)).astype('int')
		imgs = np.repeat(imgs, repeats=3, axis=3) # grayscale image

		return imgs

		
	def get_gaudy_images(self, imgs_raw, sigma=0):
		# generate gaudy images given natural images
		#
		# INPUT:
		#	imgs_raw: (num_images, num_pixels, num_pixels, 3), natural raw images (pixel intensities between 0 and 255)
		#	sigma: (float, >= 0), smoothing constant for Gaussian kernel, larger sigma --> smoother
		#		NOTE: Smoothing occurs before the gaudy transformation!
		#
		# OUTPUT:
		#	imgs_gaudy: (num_images, num_pixels, num_pixels, 3), gaudy images (pixel intensities between 0 and 255)

		num_imgs = imgs_raw.shape[0]

		imgs_gaudy = np.copy(imgs_raw)

		for iimg in range(num_imgs):
			img = imgs_gaudy[iimg]

			if sigma > 0:
				img = ndimage.gaussian_filter(img, sigma=(sigma,sigma,0))

			for ichannel in range(3):
				m = np.mean(img[:,:,ichannel])
				img[:,:,ichannel][img[:,:,ichannel] < m] = 0  # saturate colors based on average intensity for that color
				img[:,:,ichannel][img[:,:,ichannel] >= m] = 255

				imgs_gaudy[iimg] = img

		return imgs_gaudy


	def resize_images(self, imgs_raw, resize_shape=(224,224)):
		# resizes images
		#
		# INPUT:
		#	imgs_raw: (num_images, num_pixels, num_pixels, 3), raw images (pixel intensities between 0 and 255)
		#	resize_shape: [num_pixels_new,num_pixels_new], pixel height and width of input image (112 x 112 for compact model)
		#
		# OUTPUT:
		#	imgs_resized: (num_images, num_pixels_new, num_pixels_new, 3), resized raw images (pixel intensities between 0 and 255)

		imgs = np.copy(imgs_raw)

		imgs_resized = []
		for iimg in range(imgs.shape[0]):

			imgs_resized.append(np.array(Image.fromarray(imgs[iimg].astype('uint8')).resize(resize_shape)))
			
		imgs_resized = np.array(imgs_resized)

		return imgs_resized


