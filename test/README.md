This directory contains unit tests for the main project. 

The test `test_experiment.py` runs all the experiments to ensure working. It uses the Mini-SMD dataset with less epochs to minimise runtime.  
Each experiment test will be named `Test_<experiment_path>` with `_` as the separator. eg: `Test_generative_gan_train_beatgan`
