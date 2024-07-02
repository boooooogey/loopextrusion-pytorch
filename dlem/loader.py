"""Loader function for dlem models
"""
import importlib

def load_model(model_name:str):
    """Returns models from model directory.

    Args:
        model_name (str): name of the file that hosts the model.

    Returns:
        model class
    """
    module = importlib.import_module(f'dlem.models.{model_name}')
    return module.DLEM

def load_reader(reader_name:str):
    """Returns readers from reader directory.

    Args:
        model_name (str): name of the file that hosts the model.

    Returns:
        model class
    """
    module = importlib.import_module(f'dlem.readers.{reader_name}')
    return module.DLEMDataset