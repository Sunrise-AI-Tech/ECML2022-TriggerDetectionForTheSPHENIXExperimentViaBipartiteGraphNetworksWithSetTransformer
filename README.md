# Bipartite Neural Network

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
