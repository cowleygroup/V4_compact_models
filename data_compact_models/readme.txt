These folders contain the models/weights of the compact models.

Each compact model is for a single V4 neuron. We had 219 V4 neurons across
4 recording sessions (held out from training the ensemble model), so we have
219 compact models, indexed by session ID and neuron index.

The compact model was identified by training on predicted responses of the
ensemble model to 12 million images. To get the predicted responses of the ensemble
model, we took half of the images in the test
sessions to map the ensemble model embeddings to the V4 neurons. We use
the remaining half of the images for evaluating test performance.
Thus, the output response of a compact model directly maps to a single V4 neuron's
response. In other words, you can treat the 219 compact models as 219 V4 neurons.

Note: Some compact models will output negative responses to a small number of images.
This is likely due to model mismatch. In general, we ignore a compact model's overall
mean and standard deviation; we show that the "wiggles" of the compact model match the
"wiggles" of the V4 neuron across images.

architecture:
The input image batch is (num_images,112,112,3). Each compact model has 5 convolutional 
layers (layer 0 is full conv, the rest are separable convolutions) with layers 1 and 2 
having stride 2. Each layer may have a different number of filters, as determined by
the pruning step. The last layer is a dense layer that represents a spatial readout, as it
linearly combines across filters and spatial information.
An illustrative diagram of the model can be found in
Ext. Data Fig. 1 in Cowley et al., bioRxiv 2023.

You can see the Keras model summary for each compact model in ./model_summaries.
We also saved the numbers of filters across layers for each compact model in ./nums_filters.

For training the compact models, see ../code_distill_and_prune/

Here we save the compact models and their weights with multiple versions for easy access.
The most common types will be ./models_keras and ./models_torch. The rest of our code
is in Keras/tensorflow.

See ../script3_load_compact_models.py to load the compact models in various ways.