from setuptools import setup

setup(
    name='TimeSeAD',
    packages=['timesead', 'timesead_experiments'],
    version='0.0.1',
    license='MIT',
    description='TimeSeAD - Library for Benchmarking Multivariate Time Series Anomaly Detection',
    author='Dennis Wagner, Tobias Michels, Florian C.F. Schulz, Arjun Nair',
    author_email='dwagner@cs.uni-kl.de, tmichels@cs.uni-kl.de, florian.cf.schulz@tu-berlin.de, naira@rptu.de',
    url='https://github.com/wagner-d/TimeSeAD',
    keywords=[
        'time series anomaly detection',
        'anomaly detection',
        'time series',
        'benchmark'
    ],
    install_requires=[
        'matplotlib>=3.3',
        'numpy>=1.20',
        'scikit-learn>=0.24.2',
        'tqdm>=4.59',
        'pandas>=1.2',
        'scikit-learn>=0.24',
        'torch>=1.9,<2',
        'torchvision>=0.2',
        'torch-geometric',
    ],
    extras_require={
        'experiments': [
            'pymongo>=3.12',
            'sacred>=0.8',
        ],
        'docs': [
            'sphinx>=6.1',
            'sphinx-autoapi>=2.1',
            'myst-parser>=1.0',
        ]
    }
)
