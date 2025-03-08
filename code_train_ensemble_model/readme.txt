
This code will train an ensemble model on neural data and evaluate its predictions on held-out image/response pairs.

The ensemble model has the following architecture:
image --> ResNet50 (middle layer) --> ensemble member (x 25) --> average over ensemble members --> predicted responses

- for an ensemble member's architecture, see the keras summary in ./data_ensemble_model/ensemble_model_summary.txt
- each ensemble member is trained separately; at inference, we average the predicted responses across the ensemble.


script_train_ensemble_model.py:
Script to train the ensemble model on all training sessions.
- hyperparameters were found with exhaustive search (default in ensemble model class)
- saves performance for each training session on validation set (800 images pepe)
- performance measured with unbiased CV ridge regression (for final performance, use alternating factorized linear mapping method)
- epoch/session with largest validation CV performance is saved (early stopping)--> final model
- saves model for any change in the largest validation performance


script_predict_natural_images.py:
# Script to predict V4 responses in test sessions with ensemble model.
# - uses either ridge regression or factorized linear mapping


NOTE:
Training the ensemble model takes some time, and for a new dataset, new hyperparameters will likely be needed. We also found that a small amount of learning rate decay coupled with more passes per recording session across epochs improved performance. Much of the heavy lifting is in ./classes/class_ensemble_model.py.