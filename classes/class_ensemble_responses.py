
# Written by B. Cowley, 2025

# class to compute ensemble responses to 12M images (as a teacher student)

import numpy as np

class EnsembleResponseClass:

	def __init__(self, isession=0, ilarge_zip=0):

		self.isession = isession

		response_path = './ensemble_responses_for_distillation/responses_500kimages/session{:d}_zip{:d}.npy'.format(self.isession, ilarge_zip)
		self.responses = np.load(response_path)
		self.num_neurons = self.responses.shape[0]

		

	def get_responses(self, ineuron, inds):

		return self.responses[ineuron,inds]


	def get_responses_to_test_images(self, ineuron, num_test_images=10000):

		response_path = './ensemble_responses_for_distillation/responses_10k_testimages/session{:d}.npy'.format(self.isession)

		responses_test = np.load(response_path)

		return responses_test[ineuron,:num_test_images]


	def get_responses_to_realdatatest_session(self, ineuron):

		response_path = './ensemble_responses_for_distillation/responses_realdata/responses_test_session{:d}.npy'.format(self.isession)

		responses_test = np.load(response_path)

		return responses_test[ineuron,:]



