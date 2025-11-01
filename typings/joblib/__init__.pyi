from typing import Union, Dict
import os
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

def dump(
        value: Dict[str, Union[KMeans, NDArray[np.int_]]],
        filename: Union[str, os.PathLike[str]]
) -> None:
    ...

def load(
        filename: Union[str, os.PathLike[str]]
) -> Dict[str, Union[KMeans, NDArray[np.int_]]]:
    ...