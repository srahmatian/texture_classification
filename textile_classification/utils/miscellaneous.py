from pathlib import Path
from typing import Union, Tuple, List
import json
import pickle
import numpy as np
import random
import torch

def load_from_jason(filepath: Union[str, Path]) -> dict:
    """
    Load a dictionary from a JSON's filepath.

    Args:
        filepath (Union[str, Path]): the full path to the jason file you want to load.

    Returns:
        dict: loaded file in dictionary format.
    """
    with open(filepath, "r") as fp:
        d = json.load(fp)
    return d

def save_to_jason(d: dict, filepath: Union[str, Path]) -> None:
    """
    Save a dictionary to a specific location in jason format.

    Args:
        d (dict): _description_
        filepath (Union[str, Path]): _description_
    """
    with open(filepath, "w") as fp:
        json.dump(d, indent=4, fp=fp)

def load_from_pickle(filepath: Union[str, Path]) -> object:
    """
    Load from a pkl's filepath.
    it is important to note that pickled objects are not secure, 
    and you should not use pickle to load data you are not sure about its content.

    Args:
        filepath (Union[str, Path]): _description_

    Returns:
        object: _description_
    """
    with open(filepath, "rb") as fp:
        obj = pickle.load(fp)
    return obj

def save_to_pickle(obj: object, filepath: Union[str, Path]) -> None:
    """
    Save a pickable ojbect to a specific location in pickle format.
    it is important to note that pickled objects are not secure, 
    and you should not use pickling to store sensitive data or transmit it over an untrusted network.

    Args:
        obj (object): _description_
        filepath (Union[str, Path]): _description_
    """
    with open(filepath, "wb") as fp:
        pickle.dump(obj, fp)

def set_seeds(seed: int=42) -> None:
    """
    Set seed for reproducibility.

    Args:
        seed (int, optional): _description_. Defaults to 42.
    """
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
    # Set the seed for CPU and GPU computations
    torch.manual_seed(seed)
    # Disable cuDNN optimization to deterministically select an algorithm, 
    # possibly at the cost of reduced speed.
    torch.backends.cudnn.benchmark = False
    # While disabling CUDA convolution benchmarking ensures that 
    # CUDA selects the same algorithm each time an application is run, 
    # that algorithm itself may be nondeterministic, 
    # unless use following line.
    torch.backends.cudnn.deterministic = True

def get_default_device() -> torch.device:
    """
    Pick GPU if available, else CPU

    Returns:
        torch.device:
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(tensors: List[torch.Tensor], 
              device: Union[str, torch.device]) -> List[torch.Tensor]:
    """
    Move a list or of tensors to the chosen device

    Args:
        tensors (List[torch.Tensor]): list of tensors
        device (Union[str, torch.device]): the device you wanna move the tensors to

    Returns:
        List[torch.Tensor]: list of tensors on the new device
    """
    assert isinstance(tensors, [list, tuple])
    return [x.to(device) for x in tensors]

def change_artifcat_uri():
    """
    This function looks for all the meta.yaml files in mlruns folder, and 
    adapt the artifacts location based on the new directory of the project.
    """
    pass
    

