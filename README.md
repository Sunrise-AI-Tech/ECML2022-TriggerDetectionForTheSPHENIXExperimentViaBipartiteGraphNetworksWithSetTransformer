# Graph Level Prediction Pipeline

## Essential components
- The main function is main_script/main_trigger_pred.py.
- Data Loading script is dataloaders/hits_loader.py.
- Model scripts are stored in models folder.
- Configuration files are stored in configs folder.

## How to edit configuration files?
The configuration files are consist of several parts.
 
```
model_name: Diffpool
output_dir: results/Diffpool/scaled_hits/pool2_NN_trigger_1
epochs: 500

model:
  hidden_dim: 60
  hidden_activation: Tanh
  layer_norm: True
  learning_rate: 0.0005
  lr_scheduler_decrease_rate: 0.995
  diff_pool_config:
    max_num_nodes: 400
    hidden_dim: 16
    embedding_dim: 8
    label_dim: 1
    num_layers: 3
    num_pooling: 2
    assign_hidden_dim: 8
    assign_ratio: 0.25
    linkpred: False

data:
    name: hits_loader
    input_dir: ../prepare/scaled_hits_data/trigger-event
    # input_dir: ../prepare/scaled_hits_data/nontrigger-event/NN
    n_train: 200000 
    n_valid: 10000 
    real_weight: 1
    false_weight: 1
    batch_size: 100
    n_input_dir: 2
    input_dir2: ../prepare/scaled_hits_data/nontrigger-event/NN
    random_permutation: True
    n_train2: 200000 
    n_valid2: 10000 

```

- model name is used in the main function to import the model you want to use. If you create a new model, you should specify your own model name here and change the main function according to this name.

- output_dir is the folder you always want to change if you want to keep the old results. If you run several trainings using the same output_dir, you will lose your result history.

- epochs is the number of epochs in total.

- model part defined the hyper-parameters for models. This section will be directly used in the initial function of your model Class. Make sure the names of arguments are the same as your arguments for initial functions.  
Like the example above, this is from configs/Diffpool.yaml. There's one sub-dictionary in model which is called diff_pool_config. This kind of sub-dictionary is very helpful if you want to create sub-model in your class. models/Diffpool.py and models/GNNDiffpool.py are good examples you can view.

- data is related to the data loader class. This section will be directly used to create your data loaders.   
Usually you don't have to change the structure of the dataloader if you are using the same dataset, but you can change the dataloader script or add a new data loader defined by yourself (by changing the data['name'] here), if you have special needs or the dataset is changed.  
The thing you need to pay attention to is: 
    - n_input_dir decides how many dataset folder you want to use.  
    If n_input_dir = 1, only the input_dir will be used;  
    if n_input_dir = 2, both the input_dir and input_dir2 will be used.
    - input_dir and input_dir2 should be the location of your datasets. They will be different on different servers.
    - n_train + n_valid events will be loaded from input_dir, n_train2 + n_valid2 events will be loaded from input_dir2.
    - real_weight and false_weight are designed for a weighted loss function. (weighted binary cross entropy loss)

## How to run the code?
You can simply run the code by

```
python main_script/main_trigger_pred.py configs/*****.yaml
```

You can also run the code in the background by

```
nohup python main_script/main_trigger_pred.py configs/*****.yaml > log.out &
```

## How to plug in a new model?
If you want to plug in a new model designed by yourself, you can first build your model in the models folder. Then add your model to the main function in the following part:
```
    if config['model_name'] == 'GNN_ip':
        from models.ip_GNN import IpGNN
        model = IpGNN(**config['model'])
    elif config['model_name'] == 'GNN_vp':
        from models.vp_GNN import VpGNN
        model = VpGNN(**config['model'])
    elif config['model_name'] == 'GNN_Diffpool' or config['model_name'] == 'GNN_Diffpool_trackinfo':
        from models.GNN_diffpool import GNNDiffpool
        model = GNNDiffpool(**config['model'])
    elif config['model_name'] == 'GNNPairDiffpool' or config['model_name'] == 'GNNPairDiffpool_affinityloss':
        from models.GNN_pair_diffpool import GNNPairDiffpool
        model = GNNPairDiffpool(**config['model'])
    elif config['model_name'] == 'Diffpool':
        from models.Diffpool import Diffpool
        model = Diffpool(**config['model'])
    elif config['model_name'] == 'Dense_GNN_Diffpool':
        from models.DenseGNNDiffpool import DenseGNNDiffpool
        model = DenseGNNDiffpool(**config['model'])
```

The model you build should be a pytorch class, and contains initial function, forward function and training and loss function. You can view models/Diffpool.py as an example.

You can also train different part of models separately. The way to do this is, when you define your optimizers, define the params as the parameters of a part of the model, and set up several optimizers. And you can train them separately by specifying which optimizer you want to use.

If you new model has special inputs and some different information for the loss function, there are several ways to do this:
- If the operations is very time-consuming, the best way to do it is to create a new dataset and store the features you need in the data files.
- If it's not complicated, you can add the variable in the dataloader(load_graph, initial functions of the dataloader to preload the values, and add the features in the getitem function). After these, you can access the values in the main script, do_epoch function(for data in dataloader: data['your_feature']).



## Sampler for DataLoader
In this pipeline, we provided a self-defined samplar class. It is stored in dataloaders/hits_loader.py, JetsBatchSampler Class.

Using this sampler, we can make sure that for each batch, all events can have the same number of hits or tracks. The way to do it is, we can input an array when we initiate the class. The array has the same length as the number of events. The sampler will divide the events into different batch according to this array. It will make sure the events in the same batch will have the same value in this array.