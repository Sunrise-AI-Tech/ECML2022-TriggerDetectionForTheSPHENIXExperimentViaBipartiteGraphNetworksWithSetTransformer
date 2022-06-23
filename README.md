# Bipartite Neural Network

## Training and Evaluation of the Bipartite Model
Modify the following fields in `configs/gt_track_biatt.yaml`:

 - `train_input_dir`: Path to the directory containing training data.
 - `val_input_dir`: Path to the directory containing validation data.
 - `test_input_dir`: Path to the directory containing test data.
 
Run the following command to perform training and evaluation of the Bipartite model:

```
python main_scripts/main_biatt.py
```

## Training and Evaluation of the Bipartite Model with the Physics MLP

Modify the following fields in `configs/gt_track_biatt.yaml` and ``configs/gt_track_physics.yaml`

 - `train_input_dir`: Path to the directory containing training data.
 - `val_input_dir`: Path to the directory containing validation data.
 - `test_input_dir`: Path to the directory containing test data.

Train the physics MLP. The Bipartite model will use this MLP to obtain an estimate for the transverse momentum of the tracks in the event.
```
python main_scripts/main_physics.py
```
The results will be placed in the directory specified by the `output_dir` field in `configs/gt_track_physics.yaml`. Looking at end of `out_0.log`, find the epoch that had the best validation accuracy. Then, modify the `model_physics_path` field in `configs/gt_track_biatt_physics_mlp.yaml` to the checkpoint file of the best epoch of the Physics MLP.
Finally, run the following command to perform training and evaluation of the Bipartite model using the physics MLP:

```
python main_scripts/main_biatt_physics_mlp.py
```
