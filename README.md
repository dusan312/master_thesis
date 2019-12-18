# master_thesis
Repository with all source files for my Master thesis

In order to start training, change the hyperparameters 
inside the main.py file, and then run it with optional 
the second argument "--gpu1" if you want to run it on cuda device with index 0, otherwise it will be runed on cuda device with index 1.


| Hyperparameter  | Description |
| ------------- | ------------- |
| type  | it can be 'class', 'mySS' or 'rotSS' to run Classification, My Self-Supervised or RotNet Self-Supervised training  |
| dataset  | for now, it can only be 'stl10' |
| learning_rate  | Starting learning rate  |
| lr_class  | learning rate for classification which can be runed automatically after SS part |
| experiment_name | name of the experiment which we run |
| n_epoch | number of training epochs  |
| n_epoch_class | number of classification epochs after SS part  |
| save | it can be true or false, if it's true it will save the first five Conv layers during, and after the training procedure  |
|load | true or false, if it's true it will load first five Conv layers before the beginning of training procedure|
|load_name | path to the model which could be loaded|
|load_second_part | similar with load, just for the second part of the network (binary classifier, RotNet classifier, or Normal classifier) |
|save_second_part | similar to save, also for the second part of the network|
|batch_size | batch size for the training|
|batch_class | batch size for classification training, after SS part|
|test_class | true or false, if it's true it will run classification training after SS part|
|frz | (freeze) all convolutional layers from conv1 to conv5 will be frozen during the classification training (it doesn't make any changes to SS training)|

All the tensorboard data files will be stored in /data/cvg/dusan/runs/'experiment_name' folder

Models are saved on /data/cvg/dusan/experiments/'experiment_name'/models/'model_name'-together 
with date and time when the training started
