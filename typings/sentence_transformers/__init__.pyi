from typing import List
import numpy as np
from numpy.typing import NDArray

class SentenceTransformer:
    def __init__(self, model_name_or_path: str) -> None: ...
    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> NDArray[np.floating]: ...
