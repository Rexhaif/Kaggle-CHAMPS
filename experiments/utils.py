import numpy as np # linear algebra
from sklearn import metrics
from tqdm import tqdm
from numba import jit, float32


@jit(float32(float32[:], float32[:]))
def fast_log_mae(y_true: np.ndarray, y_pred: np.ndarray):
    n = y_true.shape[0]
    return np.log(np.sum(np.absolute(y_true - y_pred))/n)

def fast_metric(y_true, y_pred, types, verbose=True):
    if verbose:
        iterator = lambda x: tqdm(x)
    else:
        iterator = list
    
    per_type_data = {
        t : {
            'true': [],
            'pred': []
        } 
        for t in list(set(types))
    }
    for true, pred, t in iterator(zip(y_true, y_pred, types)):
        per_type_data[t]['true'].append(true)
        per_type_data[t]['pred'].append(pred)
        
    maes = []
    for t in iterator(set(types)):
        maes.append(
            fast_log_mae(
                np.array(per_type_data[t]['true'], dtype=np.float32),
                np.array(per_type_data[t]['pred'], dtype=np.float32)
            )
        )
    return np.mean(maes)