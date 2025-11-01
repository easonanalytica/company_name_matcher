from typing import Union, Dict, Optional, TypedDict
import os
import numpy as np
from numpy.typing import NDArray

class KMeans():
    def __init__(self, n_clusters:int) -> None:
         ...

    def fit_predict(self, X: NDArray[np.floating]) -> NDArray[np.signedinteger]: 
        ...


class LoadData(TypedDict):
    kmeans: KMeans
    clusters: NDArray[np.signedinteger]

def dump(
        value: Dict[str, Optional[Union[KMeans, NDArray[np.signedinteger]]]],
        filename: Union[str, os.PathLike[str]]
) -> None:
    ...

def load(
        filename: Union[str, os.PathLike[str]]
) -> LoadData:
    ...