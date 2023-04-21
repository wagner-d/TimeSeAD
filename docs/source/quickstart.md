```{currentmodule} timesead
```

# Quickstart Guide

This guide covers the common use cases of the TimeSeAD library and details the general structure and interface.

## Adding datasets

A dataset in TimeSeAD consists of two parts:
* A directory in `data` named after the dataset containing the raw dataset files.
* A module in {mod}`timesead.data` named `<dataset name in lower case>_dataset` implementing the dataset class named `<dataset name in camel case>Dataset`. This class is responsible for loading data from disk and making it available in the correct format.

A dataset should implement the {class}`~data.BaseTSDataset` interface.
The dataset should also provide sufficient functionality for obtaining all necessary files. 
If the dataset is readily downloadable, a download flag should be supported in the constructor. 
Otherwise, provide the necessary information to obtain the dataset in the docstrings, see {class}`~data.SWaTDataset` for an example.

We often need to apply one or more transformations to the dataset, such as windowing, subsampling, or providing additional information in the labels.
In TimeSeAD, such transformations are implemented in {mod}`timesead.data.transforms`. 
Each transformation implements the {class}`~data.transforms.Transform` interface.
All transforms that are to be applied to a dataset are collected in a pipeline, which can be specified as an ordered dictionary.
A default pipeline needs to be specified as part of the {class}`~data.BaseTSDataset` interface.

The README in the `data` directory contains a table detailing the statistics of each dataset implemented in TimeSeAD.
To compute the statistics and generate the corresponding plots of a new dataset, you can run `scripts/generate_dataset_statistics.py` with the appropriate parameters.
Running the script creates a directory for the dataset in `resources/datasets` containing a train and test directory, which contain the generated images and statistics for train and test set respectively.
To add the statistics and plots to the README, run `scripts/update_dataset_readme.py`.

## Adding methods

TimeSeAD separates each method into two components: the model and the anomaly detector.
The model specifies the architecture of a method and the anomaly detector specifies the computation of the anomaly score based on a trained model.
Some methods require the anomaly detector to be updated after training of the model is completed, for example to compute the mean of the predictions.
Furthermore, this split allows for testing with different anomaly detectors without the need to retrain the model.
For more details on anomaly detectors, refer to the documentation of {class}`~models.common.AnomalyDetector`.

Each method is implemented in a separate subpackage of {mod}`timesead.models`.
Methods are categorized by their approach to the anomaly detection problem:
* {mod}`timesead.models.generative` contains generative models such as GAN- and VAE-based models.
* {mod}`timesead.models.reconstruction` contains reconstruction-base models such as AE-based methods.
* {mod}`timesead.models.prediction` contains prediction-based models.
* {mod}`timesead.models.baselines` contains shallow baselines.
* {mod}`timesead.models.other` contains models not suited for any of the other categories such as models combining multiple approaches.
A new method should be implemented in a single file in one of these directories.

To aid implementations, the {mod}`timesead.models` package contains several common building blocks used across multiple methods:
* {mod}`timesead.models.layers` contains neural network layers such as causal convolutions.
* {mod}`timesead.models.common` contains common architectural elements such as RNNs, AEs, VAEs, and GANs, and some common anomaly detectors.

To implement a new method
1. Specify the model in a class that extends {class}`~models.BaseModel` (which in turn extends {class}`torch.nn.Module`).
2. If the model needs different optimizers for different parts, overwrite the {meth}`~models.BaseModel.grouped_parameters` method to return a tuple containing each part's parameters.
3. Implement an anomaly detector that extends {class}`~models.common.anomaly_detector.AnomalyDetector`. Some common anomaly detectors are implemented in {mod}`timesead.models.common.anomaly_detector`.
If the method requires specific preprocessing of the data, refer to the previous section detailing data transforms.

To train a method, we usually need to specify an objective often a loss to be minimized and sometimes the updates made to the model.
If a method requires a loss not already implemented in TimeSaAD or pytorch, TimeSeAD provides the {class}`~optim.loss.Loss` interface.
A loss should take in the outputs of a model and compute a scalar value.
If a method requires specialized updates during training, we can implement a custom trainer by extending the existing {class}`~optim.trainer.Trainer`.
The trainer is usually constructed by passing training and validation sets as {class}`torch.utils.data.DataLoader`, optimizer and scheduler as {class}`~typing.Callable` constructing the respective objects when called, and the device to be used in pytorch notation.
For details on optimizers and schedulers, refer to the [documentation of pytorch](https://pytorch.org/docs/stable/optim.html).
A trainer loops over the specified range of epochs for each of which it calls its {meth}`~optim.trainer.Trainer.train_epoch` method.
For each epoch, the trainer loops over all batches in the dataset and calls the {meth}`~optim.trainer.Trainer.train_batch` method for each batch.
The trainer passes a list of losses and optimizers (one each for each group of parameters returned by {meth}`~models.BaseModel.grouped_parameters`) down to {meth}`~optim.trainer.Trainer.train_batch`, where the losses are computed on a single batch and the parameters of the model are updated according to the order the losses and optimizers appear in the lists.
The default {meth}`~optim.trainer.Trainer.train_epoch` handles the updates of the schedulers (possibly one for each optimizer) and logging.
The default {meth}`~optim.trainer.Trainer.train` is mostly concerned with setting everything up for training.
We can overwrite the trainer at any level to individually adapt the trainer to the needs of a method.

## Running Evaluations

TimeSeAD is not only a collection of datasets, methods, and evaluation tools, but also provides tools for optimization.
Thus, the implemented methods can be used in any existing evaluation framework, trained and evaluated using the provided tools manually, or trained and evaluated using the `timesead_experiments` extension provided alongside the library.
In this section we will detail how the optimization tools of TimeSeAD can be used manually and how the `timesead_experiments` can be used with existing and new methods.

### Manual evaluations

To manually train a method
1. Construct the model.
2. Construct a {class}`~data.transforms.PipelineDataset`. Make sure all the necessary file for preprocessing the dataset are accounted for.
3. Create a trainer.
4. Add desired training hooks ({class}`~optim.trainer.EarlyStoppingHook`, {class}`~optim.trainer.CheckpointHook`).
5. Train the model.
6. Create an anomaly detector.
7. Train the anomaly detector if necessary.
8. Perform evaluations.

To aid evaluations, TimeSeAD provides the {class}`~evaluation.Evaluator` class.
It provides implementations of various evaluation measures including AUC, AUPRC, AP, F1, and various measure based on (recall-consistent) time series precision and recall.
To visualize the results and dataset statistics, TimeSeAD provides various plotting tools in {mod}`timesead.plots`.

### Using the `timesead_experiments` extension

To use the `timesead_experiments` extension it needs to be installed properly first, refer to the installation guide for details.

The `timesead_experiments` extension provides a training experiment for each supported method in a directory structure mirroring that of {mod}`timesead.models`.
Each experiment follows the naming scheme `train_<name of the method in snake case>.py`.
An experiment exposes the hyperparameters of a method as sacred configuration variables.
Running the training experiment should train the model for exactly one set of hyperparameters and return at least a detector that can be used later for evaluation.
When executed from the command line, the parameters can be set explicitly by adding `with '<parameter>=<value>'` or by specifying the parameters in a separate yml file and adding `with path/to/file.yml`.

To test a new method, add a corresponding experiment.
We provide a template in `timesead_experiments/train_model_template.py`.
Sacred uses [ingredients](https://sacred.readthedocs.io/en/stable/ingredients.html) to define reusable configurations.
Training experiments use two types of ingredients: 
* The `data_ingredient` is used for anything related to loading and transformation of datasets. It defines the function {meth}`timesead_experiments.utils.load_dataset` that loads a dataset class and instantiates a pipeline for it and merges the default dataset-defined pipeline with user-supplied pipeline elements. The ingredient is also responsible for splitting the data into several parts (e.g., train and test set). In the end, it returns a {class}`~data.transforms.PipelineDataset` that is compatible with torch's default dataset interface.
* The `training_ingredient` is used for anything related to the main training loop. It instantiates `torch.utils.data.DataLoaders` for some given datasets as well as a user-supplied `Trainer` class, optimizer, and a loss function. Finally, it calls the trainer's main training routine with the parameters supplied by the user as part of the ingredients configuration. It returns the trainer instance after training has completed.
Their implementations and any other technical functions can be found in `timesead_experiments/utils`.
There, both ingredients define a default configuration.
Any parameter can be overwritten by specifying that parameter in the corresponding configuration in the experiment, namely in {meth}`timesead_experiments.utils.dataset_ingredient.data_config` and {meth}`timesead_experiments.utils.training_ingredient.training_config`.
Other model specific parameters can be defined in the experiment configuration in {meth}`timesead_experiments.train_model_template.config`.
The data pipelines for both training and testing can be specified in the respective functions.
When running an experiment, the main method collects all parameters, constructs the model, constructs a trainer, sets desired training hooks, trains the model, and constructs the anomaly detector.
To ensure compatibility with the grid search experiments, an experiment should return a dictionary containing at least the model and the trainer.

To evaluate multiple configurations of one experiment, `timesead_experiments` provides an experiment for performing grid search in `timesead_experiments/grid_search.py`.
The configuration of an experiment run with grid search, are specified in yml files, which are provided in `experiment_configs`.
Alongside configurations for experiments on each dataset, this directory contains a collection of recon experiments, which can be used to estimate the total runtime of an experiment.
In such a configuration file, we can specify the training experiment, dataset, and evaluation metrics.
For each parameter of a method, we can specify a list of values to be used during grid search.

Executing any experiment will create a corresponding directory in `log`, where all results and logs will be stored.

## Plotting

TimeSeAD provides various tools for plotting time series data, which can be found in the {mod}`timesead.plots` package.
By default, all plots use the style specified in `resources/style/timesead.mplstyle`.
To change the style, refer to the relevant [matplotlib documentation](https://matplotlib.org/stable/tutorials/introductory/customizing.html).
