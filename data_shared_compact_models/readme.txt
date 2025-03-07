
These shared compact models correspond to Figure 3d in Cowley et al., bioRxiv 2023.

Shared compact models are trained on all 219 V4 neurons from 4 test sessions. They vary
in the number of convolutional filters in the first 3 layers (the remaining layers have 
100 filters): 5, 10, 25, 35, 50, 75, 100, 200 filters. They are trained either with distillation (i.e., as student models from some teacher model) or via direct fitting on V4 data.

The seven types are:
from_ensemble_model: teacher model is the ensemble model (predicted responses to 12 million images)
linear_from_ensemble_model: same as from_ensemble_model except shared compact model has no relus
from_V4data_direct_fit: model trained entirely on V4 data in similar process as ensemble model
from_resnet50: teacher model is ResNet50 (middle layer)
from_resnet50_robust: teacher model is ResNet50_robust (middle layer)
from_vgg19: teacher model is VGG19 (middle layer)
from_cornets: teacher model is CORnet-S (middle layer)

The input is a batch of images (num_images x 112 x 112 x 3) (they need to be re-centered).
The output is a matrix of responses (219 neurons x num_images).

Run script5_compute_R2s_shared_compact_models.py to see how R2s compare (reproducing Fig. 3d from our paper).
