from typing import Union, Dict, Optional
import os
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

def dump(
        value: Dict[str, Optional[Union[KMeans, NDArray[np.signedinteger]]]],
        filename: Union[str, os.PathLike[str]]
) -> None:
    ...

def load(
        filename: Union[str, os.PathLike[str]]
) -> Dict[str, Union[KMeans, NDArray[np.signedinteger]]]:
    ...