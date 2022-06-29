# Bipartite Neural Network

## Dependencies
```
conda create -n biatt cudatoolkit=11.3 numpy pandas python pytorch pytorch-scatter pyyaml scikit-learn tqdm pyg -c pyg -c pytorch
pip install -r requirements.txt
```

## Specifying the location of Training, Evaluation, and Test Data
Each model `X` has a respective file configuration file located at`configs/gt_track_X.yaml`. For example, the Bipartite model can
be run using `main_scripts/main_biatt.py` and has the configuration file `configs/gt_track_biatt.yaml`.

Modify the following fields in `configs/gt_track_biatt.yaml` to specify the location of the training, validation, and test data:

 - `train_input_dir`: Path to the directory containing training data.
 - `val_input_dir`: Path to the directory containing validation data.
 - `test_input_dir`: Path to the directory containing test data.
 
## Specifying the location of Training results.
Each configuration file has an `output_dir` field which a directory in which the results are placed. The results are placed
in a subdirectory in the directory specified by the `output_dir` field. The subdirectory has name `experiment-YYYY-MM-DD_HH:MM:SS`,
where `YYY-MM-DD_HH:MM:SS` specifies the time in which the script training the model was invoked.
_ 
## Training and Evaluation of the Bipartite Model

Specify the Bipartite model by modifying the `layers_spec` field in `configs/gt_track_biatt.yaml`. 
The `layers_spec` field is a list where each entry in the list defines a layer in the model.
Each entry has the format `(hidden_dimension, num_aggregators)`.
Thus, `layers_spec: [[64, 16], [128, 16]]` specifies a bipartite model in which the first layer has a hidden dimension of `64` and there are `16` aggregators, and the final layer has a hidden dimension of `128` and has `64` aggregators.

Run the following command to perform training and evaluation of the Bipartite model:

```
python main_scripts/main_biatt.py
```

## Training and Evaluation of the Bipartite Model with the Physics MLP
Specify the Physics MLP  by modifying the `n_hidden`, `hidden_size`, and `hidden_activation` fields in `configs/gt_track_phsyics.yaml`. 
The `layers_spec` field is a list where each entry in the list defines a layer in the model. The `n_hidden` field denotes how many hidden layers to apply. The `hidden_size` field denotes the size of each hidden layer. The `hidden_activation` field denotes the activation function used in the hidden layers.
Thus the following model configuration denotes an MLP with three hidden layers, each with a hidden size of `128`, and using the ReLU activation function.
```
n_hidden: 3
hidden_size: 128
hidden_activation: ReLU
```


Train the physics MLP. The Bipartite model will use this MLP to obtain an estimate for the transverse momentum of the tracks in the event.
```
python main_scripts/main_physics.py
```
The results will be placed in the directory specified by the `output_dir` field in `configs/gt_track_physics.yaml`. Looking at end of `out_0.log`, find the epoch that had the best validation accuracy. Then, modify the `model_physics_path` field in `configs/gt_track_biatt_physics_mlp.yaml` to the checkpoint file of the best epoch of the Physics MLP.

Specify the Bipartite model by modifying the `layers_spec` field in `configs/gt_track_biatt.yaml`. 
The `layers_spec` field is a list where each entry in the list defines a layer in the model.
Each entry has the format `(hidden_dimension, num_aggregators)`.
Thus, `layers_spec: [[64, 16], [128, 16]]` specifies a bipartite model in which the first layer has a hidden dimension of `64` and there are `16` aggregators, and the final layer has a hidden dimension of `128` and has `64` aggregators.

Finally, run the following command to perform training and evaluation of the Bipartite model using the physics MLP:

```
python main_scripts/main_biatt_physics_mlp.py
```

## Training and Evalution of Set Transformer
Specify the Set Transformer model by modifying the following fields in `configs/gt_track_settrans.yaml`:

 - `num_inds`: The number of inducing points used in each set attention block.
 - `dim_hidden`: The dimension of fully connected layers used in the hidden layers of the Set Transformer.
 - `num_heads`: The number of heads used in each set attention block.
 - `ln`: Specifies whether to apply a layer norm after each set attention block.
 
 
Run the following command to perform training and evaluation of Set Transformer:

```
python main_scripts/main_settrans.py
```

## Training and Evaluation of ParticleNet+SagPool
Specify the ParticleNet model by modifying the following fields in `configs/gt_track_particlenet.yaml` and in `configs/gt_track_sagpool.yaml`:

 - `k`: The number of nearest neighbors to use in the edge convolution blocks.
 - `hidden_dim`: The dimension of fully connected layers used in each ParticleNet layer
 - `layer_norm`: Specifies whether to apply layer normalization in the ParticleNet layers.
 - `affinity_loss_CE_weight`: Specifies the weight of the cross-entropy loss for the affinity matrix and ground-truth matrix.
 - `affinity_loss_Lp_weight`: Specifies the weight of the laplacian loss for the affinity matrix.
 - `affinity_loss_11_weight`: Specifies the weight of the L\_11 loss of the affinity matrix.
 - `affinity_loss_frobenius_weight`: Specifies the weight of the frobenius loss of the affinity matrix.
 
Train the ParticleNet model. The Sagpool model will use this model to obtain an a predicted affinity matrix for each event.
The results will be placed in the directory specified by the `output_dir` field in `configs/gt_track_particlenet.yaml`. Looking at end of `out_0.log`, find the epoch that had the best validation accuracy. Then, modify the `checkpoint_file_particlenet` field in `configs/gt_track_sagpool.yaml` to the checkpoint file of the best epoch of the ParticleNet model.

```
python main_scripts/main_particlenet.py
```
Specify the SagPool model using the following fields in `config/gt_track_sagpool.yaml`. 

 - `is_hierarchical`: Specifies whether to perform hierarchical pooling.
 - `nhid`: Specifies the dimensions of fully connected layers used in each Sagpool layer.
 - `pooling_ratio`: Specifies the Sagpool pooling ratio.
 - `dropout_ratio`: Specifies the dropout ratio used in dropout layers in the model.
 
