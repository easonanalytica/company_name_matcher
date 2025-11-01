# typings/h5py/__init__.pyi
from typing import Any, Union, Tuple, Optional, Type
import numpy as np
from numpy.typing import NDArray
from types import TracebackType

class Dataset:
    shape: Tuple[int, ...]
    dtype: Any

    def __getitem__(self, key: Union[int, str, slice, Tuple[Union[int, slice], ...]]) -> NDArray[np.floating]:
        ...

    def __setitem__(self, key: Union[int, str, slice, Tuple[Union[int, slice], ...]], value: NDArray[np.floating]) -> None:
        ...

class File:
    def __init__(self, name: str, mode: str = 'r') -> None: ...
    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> Optional[bool]: ...
    def create_dataset(
        self, 
        name: str, 
        shape: Union[Tuple[int, ...], None] = None, 
        data: Union[NDArray[np.floating], None] = None,
        dtype: Any = None,
        compression: Union[str, None] = None
    ) -> Dataset: ...
    def __getitem__(self, key: str) -> Dataset: ...
    def __enter__(self) -> File: ...

def special_dtype(vlen: Any = None) -> Any: ...
