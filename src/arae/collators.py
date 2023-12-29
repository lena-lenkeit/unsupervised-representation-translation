from typing import Any, Dict, List

import numpy as np
import optree
import torch
from optree import PyTree


# Define a custom collator
class PyTreeCollator:
    """Collates leaves of a list of PyTrees with identical structure into batches"""

    def __call__(self, features: List[PyTree]) -> PyTree:
        def map_fn(x: Any, *xs: Any) -> Any:
            if isinstance(x, torch.Tensor):
                y = torch.stack((x,) + xs, dim=0)
            elif isinstance(x, np.ndarray):
                y = np.stack((x,) + xs, axis=0)
                y = torch.from_numpy(y)
            elif isinstance(x, (float, int, bool)):
                y = np.asarray((x,) + xs)
                y = torch.from_numpy(y)
            else:
                raise TypeError(x)

            return y

        return optree.tree_map(map_fn, features[0], *features[1:])
