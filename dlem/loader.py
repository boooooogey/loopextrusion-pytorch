"""Loader function for dlem models
"""
import importlib
import importlib.util
import os

def import_class_from_file(file_path:str, class_name:str):
    """
    Dynamically imports a class from a given file.

    Args:
        param file_path (str): Path to the Python file
        param class_name (str): Name of the class to import

    Returns:
        The imported class
    """
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, class_name)
    return cls

def load_model(model_name:str, import_from_file:bool=False, import_class_name:str="DLEM"):
    """
    Returns models from the model directory.

    Args:
        model_name (str): Name of the file that hosts the model.
        import_from_file (bool): Alternatively reads from a file outside of the package folder.
        Defaults to False.
        import_class_name (str): Name of the class to import. Defaults to "DLEM".

    Returns:
        The model class
    """
    if import_from_file:
        return import_class_from_file(model_name, import_class_name)
    else:
        module = importlib.import_module(f'dlem.models.{model_name}')
        return getattr(module, import_class_name)

def load_reader(reader_name:str, import_from_file:bool=False):
    """Returns readers from reader directory.

    Args:
        model_name (str): name of the file that hosts the model.
        import_from (bool): alternatively reads from a file outside of the package folder.

    Returns:
        model class
    """
    if import_from_file:
        return import_class_from_file(reader_name, "DLEMDataset")
    else:
        module = importlib.import_module(f'dlem.readers.{reader_name}')
        return module.DLEMDataset

def load_class_from_file(file_path: str, class_name: str):
    # Ensure the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")

    # Create a module specification from the file path
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    
    # Create a module from the specification
    module = importlib.util.module_from_spec(spec)
    
    # Load the module
    spec.loader.exec_module(module)
    
    # Get the class from the module
    cls = getattr(module, class_name)
    
    return cls

def load_trunk(trunk_name:str):
    """Returns a trunk from trunk directory.

    Args:
        model_name (str): name of the file that hosts the model.
        import_from (bool): alternatively reads from a file outside of the package folder.

    Returns:
        model class
    """
    module = importlib.import_module('dlem.trunks.seq_pooler')
    cls = getattr(module, trunk_name)
    return cls

def load_head(head_name:str):
    """Returns a head from head directory.

    Args:
        model_name (str): name of the file that hosts the model.
        import_from (bool): alternatively reads from a file outside of the package folder.

    Returns:
        model class
    """
    module = importlib.import_module('dlem.trunks.seq_pooler')
    cls = getattr(module, head_name)
    return cls