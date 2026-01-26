"""
Convert SMPL pkl file from chumpy format to chumpy-free format.
Run this with opencap environment.
"""

import pickle
import numpy as np
from scipy.sparse import csc_matrix

def convert_to_numpy(obj):
    """Convert chumpy object to numpy array."""
    if hasattr(obj, 'r'):  # chumpy object has .r attribute for the actual value
        return np.array(obj.r)
    elif isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, csc_matrix):
        return obj  # keep sparse matrix as is
    else:
        return obj

def main():
    input_path = '/root/TVB/SMPL2AddBiomechanics/models/smpl/SMPL_NEUTRAL.pkl'
    output_path = '/root/TVB/MHRtoSMPL/SMPL_NEUTRAL_chumpy_free.pkl'

    print(f'Loading {input_path}...')
    with open(input_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    print('Converting chumpy objects to numpy...')
    new_data = {}
    for k, v in data.items():
        type_name = type(v).__module__ + '.' + type(v).__name__
        if 'chumpy' in type_name.lower():
            print(f'  Converting {k} from chumpy to numpy')
            new_data[k] = convert_to_numpy(v)
        else:
            new_data[k] = v

    print(f'Saving to {output_path}...')
    with open(output_path, 'wb') as f:
        pickle.dump(new_data, f)

    print('Done!')

    # Verify
    print('\nVerifying...')
    with open(output_path, 'rb') as f:
        verify_data = pickle.load(f)

    for k, v in verify_data.items():
        type_name = type(v).__module__ + '.' + type(v).__name__
        print(f'  {k}: {type_name}')

if __name__ == '__main__':
    main()
