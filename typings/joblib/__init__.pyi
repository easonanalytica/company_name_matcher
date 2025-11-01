from typing import Union, Dict, Optional, TypedDict
import os
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans as SKLearnKMeans


class LoadData(TypedDict):
    kmeans: SKLearnKMeans
    clusters: NDArray[np.signedinteger]

def dump(
        value: Dict[str, Optional[Union[SKLearnKMeans, NDArray[np.signedinteger]]]],
        filename: Union[str, os.PathLike[str]]
) -> None:
    ...

def load(
        filename: Union[str, os.PathLike[str]]
) -> LoadData:
    ...