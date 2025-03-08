# V4_compact_models
Data, models, and code for the paper "Compact deep neural network models of visual cortex".

This website is intended for researchers interested in computational models of visual cortical neurons (in this case primate V4). Please post any issues/questions in Issues that we may better clarify.

We plan to release a user-friendly website that allows researcher and non-researchers (high school students, undergrads) to explore V4 neurons and deep neural network models:
[link here soon]
Users will be able to view preferred stimuli, explore filter weights, synthesize images, follow tutorials, and run their own Colab experiments on the compact models. Stay tuned!

# Data
The processed V4 responses and images are stored in ./data_V4_responses.
You can access the raw spike recordings and other raw data here: [link coming soon]

# Models
The ensemble model is stored in ./data_ensemble_model/
The compact models are stored in ./data_compact_models/
The shared compact models are stored in ./data_shared_compact_models/
Each has a readme.txt to describe the models as well as Keras model summaries.
The models were trained in Keras/tensorflow; we also provide a PyTorch version for the compact models.

# Code
The code for training the ensemble model is in ./code_train_ensemble_model/
The code for distilling and pruning to obtain a compact model is in ./code_distill_and_prune/
./classes/ contains many useful classes for image processing, accessing neural data, and model definitions.

We include a large number of scripts (1 through 6) to load the data, load the models, obtain predicted resposnes, 
compute prediction performances, and identify synthesized images:
script1: Plot a subset of images and all V4 responses as a heatmap for a given recording session.
script2: Compute prediction performance for a task-driven DNN and the ensemble model (Fig. 1b).
script3: Load compact models with many different versions available (keras + pytorch).
script4: Compute prediction performance for compact models on real V4 responses (Fig. 1h).
script5: Compute prediction performance for shared compact models (Fig. 3d).
script6: Optimize synthesized images to maximize a compact model's output (Fig. 2a).

# Citation
If using any code, model, or data from this repository, please cite:
Cowley, B.R., Stan, P.L., Pillow, J.W. and Smith, M.A., 2023. Compact deep neural network models of visual cortex. *bioRxiv*, pp.2023-11.

# About us
This is joint work from Benjamin Cowley (CSHL), Patricia Stan (CMU), Jonathan Pillow (Princeton), and Matthew Smith (CMU). 
This work was funded by a CV Starr Fellowship, Pershing Square Innovation Fund, NIH grant (F31EY031975), Simons
Collaboration on the Global Brain Investigator Award (SCGB AWD543027), NIH BRAIN Initiative grants (NS104899 and
R01EB026946), a U19 NIH-NINDS BRAIN Initiative Award (5U19NS104648), NIH grants R01EY029250 and R01EY037194.

This repository is maintained by Benjamin Cowley (CSHL).

