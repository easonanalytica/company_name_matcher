import numpy as np
from numpy.typing import NDArray

class KMeans():
    def __init__(self, n_clusters:int) -> None:
         ...

    def fit_predict(self, X: NDArray[np.floating]) -> NDArray[np.signedinteger]: 
        ...

    def transform(self, X: NDArray[np.floating]) -> NDArray[np.floating]: 
        ...

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.signedinteger]:
        ...