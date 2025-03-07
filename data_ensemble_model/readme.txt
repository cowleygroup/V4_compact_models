
These are the 25 ensemble members of the ensemble model (trained).

architecture:
Each ensemble member has the same architecture: 4 residual layers. 
You can see the Keras summary in ./ensemble_model_summary.txt.
An illustrative diagram of the model can be found in
Ext. Data Fig. 1 in Cowley et al., bioRxiv 2023.

To train these weights, you can follow the code in:
../code_train_ensemble_model

To predict with the ensemble model, see:
../script2_compute_R2s_taskdriven_vs_ensemblemodel.py

NOTE:
Each ensemble model is trained separately. Then, during inference,
we average the predicted responses across the ensemble to compute
the final predicted response.
For each recording session, we linearly map the ensemble model's
embeddings to V4 responses either with ridge regression (fast) or
factorized linear mapping (slower).


