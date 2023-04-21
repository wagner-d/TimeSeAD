# Welcome to TimeSeAD's documentation!

**TimeSeAD** is a library for developing and evaluating time series anomaly detection methods with focus on multivariate 
data and includes several datasets, methods, and evaluation tools. It was initially developed in the context of a 
paper analyzing evaluations of deep learning based methods for multivariate time series anomaly detection:

> Developing new methods for detecting anomalies in time series is of great practical significance, but progress is 
> hindered by the difficulty of assessing the benefit of new methods, for the following reasons. (1) Public benchmarks 
> are flawed (e.g., due to potentially erroneous anomaly labels), (2) there is no widely accepted standard evaluation 
> metric, and (3) evaluation protocols are mostly inconsistent. In this work, we address all three issues: (1) We 
> critically analyze several of the most widely-used multivariate datasets, identify a number of significant issues, and 
> select the best candidates for evaluation. (2) We introduce a new evaluation metric for time-series anomaly detection, 
> which—in contrast to previous metrics—is recall consistent and takes temporal correlations into account. (3) We analyze 
> and overhaul existing evaluation protocols and provide the largest benchmark of deep multivariate time-series anomaly 
> detection methods to date. We focus on deep-learning based methods and multivariate data, a common setting in modern 
> anomaly detection. We provide all implementations and analysis tools in a new comprehensive library for Time Series 
> Anomaly Detection, called TimeSeAD.

The paper can be found [here](https://openreview.net/forum?id=iMmsCI0JsS).

## Getting started
To install the TimeSeAD library please follow the instructions in {doc}`installation`.

The {doc}`quickstart` provides a short introduction on how to use and extend the library.

```{note}
This project is under active development.
```

## Contents
```{toctree}
:maxdepth: 1

installation
quickstart
```

## Citation
If you use the library in your own work, please consider citing the paper:
```
@article{
    wagner2023timesead,
    title={TimeSe{AD}: Benchmarking Deep Multivariate Time-Series Anomaly Detection},
    author={Wagner, Dennis and Michels, Tobias and Nair, Arjun and Schulz, Florian CF and Rudolph, Maja and Kloft, Marius},
    journal={Transactions on Machine Learning Research},
    year={2023},
    url={https://openreview.net/forum?id=iMmsCI0JsS},
    note={To Appear}
}
```
