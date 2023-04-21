# TimeSeAD

## How to add a new method
1. Specify the new method's architecture in a separate Python file within the `src/models` directory. The model should be specified in a class that extends `model.BaseModel` (which in turn extends `torch.nn.Module`). If your model needs different optimizers for different parts, overwrite the `grouped_parameters` methods to return a tuple containing each part's parameters.
2. Use one of the existing anomaly score implementations, e.g., `models.common.MSEReconstructionAnomalyDetector` for a reconstruction model, or create your own class that extends `models.common.AnomalyDetector`. Please see comments inside that class for more information.
3. If your method needs some sort of specific data preprocessing that is not yet implemented by one of the existing transforms, you might need to create a new subclass of `data.transforms.Transform` to use it in your pipeline later. Detailed information about what each method does can be found in the class comments.
4. If you are using timesead_experiments, create a new experiment file in the `timesead_experiments` folder to train your model. Those must define a number of methods with specific names to work with the `grid_search` experiment, for example. Hence, the best idea is to copy the `train_model_template` and change the parts where the model or anomaly detector are instantiated. You may also want to change the default pipeline and/or default parameters like optimizer, batch size, etc.

## Framework procedure
This section describes the execution flow within the framework using the example of grid search.

### Grid search
Gird search itself is a sacred experiment (`timesead_experiments/grid_search.py`), so you can call it on the command line while setting certain parameters defined in the `config` function either directly on the command line or using a config file in `experiment_configs`.
The script parses the parameter grid and executes the given training experiment (`train_*.py` in the experiments folder) for each combination of hyperparameters. It does this by spawning a configurable number of processes, so multiple instances of the training script can run in parallel on the same GPU. The training experiment trains the model and instantiates an `AnomalyDetector`, which it saves to disk.

After training models for all hyperparameter combinations, the script enters the evaluation phase. For that, we adopted a cross-validation-like procedure: The test dataset is split into $k$ folds, where we use each fold as the validation set once and the rest as test set. That means all trained detectors are loaded from disk and evaluated on the validation set. If the detector has hyperparameters too, the script will loop over those as well. We choose the detector with the best performance and record its score on the test set. Finally, we report the average score over all test folds.

Note that the training experiments need to follow a certain set of conventions to work with grid search. We will introduce those in the next section.

### Training experiments
Each TimeSeAD method needs a training experiment (`train_*.py`), that puts a `BaseModel` and an `AnomalyDetector` together and trains them. Furthermore, it exposes the method's hyperparameters as sacred configuration variables.

Running the training experiment should train the model for exactly one set of hyperparameters and return at least a detector that can be used later for evaluation. Training experiments make use of the `data_ingredient` for loading data and transforming it and the `training_ingredient` for the main training loop. We will explain those in further detail in the next sections.

To work with grid search, a training experiment should contain the following components:
* The main experiment should return a dictionary with at least `detector`, `model` and `trainer` as its keys (Dennis is currently working on this part so that it is no longer necessary to return the trainer). It should also save the model and detector in a file called `final_model.pth`.
* Functions `get_training_pipeline` and `get_test_pipeline` that return default pipelines for the model
* A function `get_batch_dim` that returns the dimension index at which batching occurs (We are planning to replace this with universal shape specifiers)
* A command `get_anomaly_detector` that fits and returns an anomaly detector for the given model.

For more information on signatures and return values of the aforementioned functions check `timesead_experiments/train_model_template.py`. It is usually a good idea to copy this file when writing a new training experiment.

### Experiment ingredients
Ingredients are sacred commands with their own configuration that can be re-used by multiple experiments. We created both a data loading ingredient and a training ingredient that most experiments use.

The data loading ingredient (`experiments/experiment_utils/dataset_ingredient.py`) defines the function `load_dataset` that loads a dataset class and instantiates a pipeline for it. It merges the default dataset-defined pipeline with user-supplied pipeline elements. The ingredient is also responsible for splitting the data into several parts (e.g., train and test set). In the end, it returns a `PipelineDataset` that is compatible with torch's default dataset interface.

The training ingredient, on the other hand, instantiates `torch.utils.data.DataLoader`s for some given datasets as well as a user-supplied `Trainer` class, optimizer, and a loss function. Finally, it calls the trainer's main training routine with the parameters supplied by the user as part of the ingredients configuration. It returns the trainer instance after training has completed. Please see the following sections for a description of the default trainer's training routine.

### Dataset loading/processing
Each dataset is implemented as a dataset class in `timesead/data`. This class is responsible for loading data from disk and making it available in the correct format. Expensive preprocessing of raw downloaded datasets that should stay the same for all methods should be performed in a separate experiment in `experiments/data`.

Data transformations that can differ from method to method should instead be implemented as part of a *pipeline*. `src/data/transforms.py` contains several of those pipeline steps (called *transforms*). Some examples include windowing, subsampling, etc. Their advantage is that users can easily change certain steps of the pipeline when running an experiment.

The pipeline is finally used in a `PipelineDataset` that asks the pipeline for a specific datapoint, which is then generated on the fly from the base dataset. 

### Training
The training ingredient instantiates a `Trainer` subclass (`timesead/optim/trainer.py`), which trains a given model on a given dataset using a given loss function and optimizer. The default trainer class trains a model for a given number of epochs on the entire training dataset. For that, it optionally shuffles the dataset before generating equally-sized batches. It then feeds each batch through the model, computes the loss and its gradient w.r.t. the model's parameters, and uses the optimizer to update those parameters. If the model has multiple parameter groups (see next section), it is also possible to specify a different optimizer and loss function for each group. The trainer computes a separate forward and backward pass and performs a parameter update for each loss in the order that they are specified in the configuration.

It is also possible to add hook modules to the trainer: By default, a checkpoint hook that saves the model to disk on a regular basis is already enabled. We also implemented early stopping via such a hook, please check `timesead/optim/trainer.py`.

Models that need a completely different training routine can create their own trainer subclass. You can override each of the trainer's methods, so if you want to change only how training with a single batch works, you can override only `train_batch`. Please see `timesead/models/generative/lstm_vae_gan.py` for an example.  

### Models
Code that defines a model's architecture should be placed under `timesead/models` in the appropriate subdirectory. The model itself should be a subclass of `BaseModel`, which in turn derives from `torch.nn.Model`. Of course, it is possible to use different building blocks within such a model. Those do not necessarily have to derive from `BaseModel`.

In case the model has different parts that must be trained separately with a different loss or optimizer, you should override the model's `grouped_parameters` method. It should return a tuple of parameter iterables, where each tuple element contains the parameters of a different part.

### Anomaly detectors
Anomaly detectors work on top of models in the sense that they take a models output and compute an anomaly score from it (some detectors also need the original input, e.g., for reconstruction-based methods). Each detector should subclass `AnomalyDetector` (`timesead/models/common/anomaly_detector.py`) and implement it's abstract methods except for `compute_offline_anomaly_score`, which is currently unused. Note that detectors which don't require any fitting of variables at training time should just implement an empty `fit` method.
