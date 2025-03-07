
Here we include the code for converting the ensemble model to a compact model.

We take as the teacher model our ensemble model. We train linear mappings between
the ensemble model's embeddings to V4 neurons in our test session (half of the data).
We then pass 12M images through this model to get the predicted V4 responses to these
12M images. We will use these (image,response) pairs to train the compact models.

Step 1: Distillation.
We run 'script_distill_train_distilled_model.py' that trains a 5 layer (100 filters/layer)
model on the ensemble responses for a given V4 neuron. 

Step 2: Pruning.
We take the distilled model and then ablate filters that contribute little to its layer's output.
This is in two steps:
2.1: 'script_prune_step1_prune_distilled_model.py' For each layer (starting with the last), filters are first ordered based on the variance of
their responses to 5k images. We then keep the top filters that explain 90% of the layer's
output. This works well for the separable convolutional layers. Note that because the first
layer's number of filters equals that of the second layer's conv. filters, we automatically 
prune the first layer by pruning the second layer.
This process is different than typical channel-wise pruning, which assesses the change in
the final response of the model.
2.2: 'script_prune_step2_retrain_with_new_network.py' After this pruning, we reset the network to the right number of filters (removing the
extraneous filters) and re-train on the ensemble responses again.

The resulting model is called a compact model. We test the compact model's performance on the real data (the remaining half of samples)
in 'script_prune_predict_step2_model.py'.