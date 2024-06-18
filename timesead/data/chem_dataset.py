from typing import List
from .generic_dataset import GenericDataset

import logging

_logger = logging.getLogger(__name__)

class ChemDataset(GenericDataset):
    """
    Dataset class to load the chemical dataset.
    """

    FEATURES = [
        "liquid mass fractions / component 1 stage 1 / [kg/kg]",
        "liquid mass fractions / component 2 stage 1 / [kg/kg]",
        "liquid mole fractions / component 1 stage 1 / [mol/mol]",
        "liquid mole fractions / component 2 stage 1 / [mol/mol]",
        "vapor mass fractions / component 1 stage 1 / [kg/kg]",
        "vapor mass fractions / component 2 stage 1 / [kg/kg]",
        "vapor mole fractions / component 1 stage 1 / [mol/mol]",
        "vapor mole fractions / component 2 stage 1 / [mol/mol]",
        "temperature / stage 1 / [K]",
        "temperature / stage 8 / [K]",
        "x apparatus molar / component 1 / [mol/mol]",
        "x apparatus molar / component 2 / [mol/mol]",
        "x apparatus mass / component 1 / [kg/kg]",
        "x apparatus mass / component 2 / [kg/kg]"
    ]

    def __init__(self, path: str, training: bool=True, standardize: bool=True,
                 preprocess: bool=True, overwrite: bool=False):
        """
        :param path: Path to the dataset
        :param training: Whether to load the training or test data
        :param standardize: Whether to standardize the data
        :param preprocess: Whether to setup the dataset for experiments
        :param overwrite: Whether to overwrite existing files
        """
        super().__init__(
            path, name=None, separator=';',
            features=self.FEATURES,
            anomaly_feature='anomaly',
            training=training, standardize=standardize,
            preprocess=preprocess, overwrite=overwrite)

    @staticmethod
    def get_feature_names() -> List[str]:
        return ChemDataset.FEATURES
