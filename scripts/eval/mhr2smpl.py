import os, glob
import numpy as np
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from mhr_smpl_conversion.mhr2smpl import mhr2smpl

from typing import Dict
import numpy as np
import torch

def dict_numpy_to_torch(
    x: Dict[str, object],
    *,
    device=None,
    dtype_map=None,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}

    for k, v in x.items():
        # ndarray
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v)
            if dtype_map and v.dtype in dtype_map:
                t = t.to(dtype=dtype_map[v.dtype])

        # numpy scalar: np.float32 / np.int64 / np.float_
        elif isinstance(v, np.generic):
            t = torch.tensor(v)   # 0-dim tensor
            if dtype_map and v.dtype in dtype_map:
                t = t.to(dtype=dtype_map[v.dtype])

        else:
            raise TypeError(
                f"Key '{k}': unsupported type {type(v)}, expected np.ndarray or np.generic"
            )

        if device is not None:
            t = t.to(device)

        out[k] = t

    return out


if __name__ == "__main__":
    root_path = "/root/autodl-tmp/outputs/downtown_arguing_00/mhr_params"
    mhr_file_list = glob.glob(os.path.join(root_path, "*data.npz"))
    mhr_file_list.sort()
    mhr_list = []
    for mhr_file in mhr_file_list:
        mhr_list.append(np.load(mhr_file, allow_pickle=True)["data"].tolist())
    
    xxx = dict_numpy_to_torch(mhr_list[0][0])
    mhr2smpl(xxx)
    a = 1