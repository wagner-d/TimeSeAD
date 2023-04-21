## Setup Environment
1. Create your Python 3 virtual environment. For [conda](https://docs.conda.io/en/latest/miniconda.html) users, `conda create -n TimeSeAD python=3` ([Docs](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands))
2. Activate the environment with `conda activate TimeSeAD`
3. Install any Pytorch version between 1.9 and 2.0 **using pip**. [Pytorch previous versions](https://pytorch.org/get-started/previous-versions/). (Pytorch 2 has some issues #63)
4. Install the TimeSeAD library. Run `pip install -e .` from within the repository.
5. (optional) To run the experiments in the library. Install with `pip install -e .[experiments]`
