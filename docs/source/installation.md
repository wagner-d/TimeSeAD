# Installation
To install the TimeSeAD library, first clone the code to your local machine:
```
git clone https://github.com/wagner-d/TimeSeAD.git
```

## Setup an environment
We recommend to set up a virtual python environment before installing TimeSeAD. The following instructions assume that you are using `conda` for this. 

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if you do not  have any `conda` installation on your system.
2. Open a terminal, `cd` into the project folder and create a new environment with
   ```
   conda env create --file setup/conda_env_cpu.yaml
   ```
   This will also install basic dependencies such as PyTorch. If you want to use TimeSeAD with NVIDIA GPU support, replace `setup/conda_env_cpu.yaml` with `setup/conda_env_cuda.yaml` (CUDA 10.2) or `setup/conda_env_cuda111.yaml` (CUDA 11.1).
3. Activate the environment with
   ```
   conda activate TimeSeAD
   ```

## Install the TimeSeAD library
We recommend installing the library in development mode so that you can make changes as you need them. This is easily achieved using `pip`:
```
pip install -e .
```
If you want to use the experiments as well, install the library as follows to include optional dependencies:
```
pip install -e .[experiments]
```

Now you can use the TimeSeAD library in your own projects by simply importing it
```python
import timesead
# Or just import certain parts
from timesead.models.prediction import TCNPrediction
```
