import numpy as np # linear algebra
from sklearn import metrics
from tqdm import tqdm
from numba import jit, float32

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection._split import check_cv
from sklearn.base import clone, is_classifier
from scipy.stats import kurtosis, skew
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

class ClassifierTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator=None, n_classes=2, cv=3):
        self.estimator = estimator
        self.n_classes = n_classes
        self.cv = cv
    
    def _get_labels(self, y):
        y_labels = np.zeros(len(y))
        y_us = np.sort(np.unique(y))
        step = int(len(y_us) / self.n_classes)
        
        for i_class in range(self.n_classes):
            if i_class + 1 == self.n_classes:
                y_labels[y >= y_us[i_class * step]] = i_class
            else:
                y_labels[
                    np.logical_and(
                        y >= y_us[i_class * step],
                        y < y_us[(i_class + 1) * step]
                    )
                ] = i_class
        return y_labels
        
    def fit(self, X, y):
        X = X.replace([np.inf,-np.inf], np.nan)
        X = X.fillna(0)
        y_labels = self._get_labels(y)
        cv = check_cv(self.cv, y_labels, classifier=is_classifier(self.estimator))
        self.estimators_ = []
        
        for train, _ in cv.split(X, y_labels):
            X = np.array(X)
            self.estimators_.append(
                clone(self.estimator).fit(X[train], y_labels[train])
            )
        return self
    
    def transform(self, X, y=None):
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        X = X.replace([np.inf,-np.inf], np.nan)
        X = X.fillna(0)
        X = np.array(X)
        X_prob = np.zeros((X.shape[0], self.n_classes))
        X_pred = np.zeros(X.shape[0])
        
        for estimator, (_, test) in zip(self.estimators_, cv.split(X)):
            X_prob[test] = estimator.predict_proba(X[test])
            X_pred[test] = estimator.predict(X[test])
        return np.hstack([X_prob, np.array([X_pred]).T])

    
class MoreStructureProperties(TransformerMixin, BaseEstimator):
    
    def __init__(self,atomic_radius,electronegativity):
        self.atomic_radius = atomic_radius
        self.electronegativity = electronegativity
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        atom_rad = [self.atomic_radius[x] for x in X['atom'].values]
        X['rad'] = atom_rad
        position = X[['x','y','z']].values
        p_temp = position
        molec_name = X['molecule_name'].values
        m_temp = molec_name
        radius = X['rad'].values
        r_temp = radius
        bond = 0
        dist_keep = 0
        dist_bond = 0 
        no_bond = 0
        dist_no_bond = 0
        dist_matrix = np.zeros((X.shape[0],2*29))
        dist_matrix_bond = np.zeros((X.shape[0],2*29))
        dist_matrix_no_bond = np.zeros((X.shape[0],2*29))
        
        for i in range(29):
            p_temp = np.roll(p_temp,-1,axis=0)
            m_temp = np.roll(m_temp,-1,axis=0)
            r_temp = np.roll(r_temp,-1,axis=0)
            mask = (m_temp==molec_name)
            dist = np.linalg.norm(position-p_temp,axis=1) * mask            
            dist_temp = np.roll(np.linalg.norm(position-p_temp,axis=1)*mask,i+1,axis=0)
            diff_radius_dist = (dist-(radius+r_temp)) * (dist<(radius+r_temp)) * mask
            diff_radius_dist_temp = np.roll(diff_radius_dist,i+1,axis=0)
            bond += (dist<(radius+r_temp)) * mask
            bond_temp = np.roll((dist<(radius+r_temp)) * mask,i+1,axis=0)
            no_bond += (dist>=(radius+r_temp)) * mask
            no_bond_temp = np.roll((dist>=(radius+r_temp)) * mask,i+1,axis=0)
            bond += bond_temp
            no_bond += no_bond_temp
            dist_keep += dist * mask
            dist_matrix[:,2*i] = dist
            dist_matrix[:,2*i+1] = dist_temp
            dist_matrix_bond[:,2*i] = dist * (dist<(radius+r_temp)) * mask
            dist_matrix_bond[:,2*i+1] = dist_temp * bond_temp
            dist_matrix_no_bond[:,2*i] = dist * (dist>(radius+r_temp)) * mask
            dist_matrix_no_bond[:,2*i+1] = dist_temp * no_bond_temp
        X['n_bonds'] = bond
        X['n_no_bonds'] = no_bond
        X['dist_mean'] = np.nanmean(np.where(dist_matrix==0,np.nan,dist_matrix), axis=1)
        X['dist_median'] = np.nanmedian(np.where(dist_matrix==0,np.nan,dist_matrix), axis=1)
        X['dist_std_bond'] = np.nanstd(np.where(dist_matrix_bond==0,np.nan,dist_matrix), axis=1)
        X['dist_mean_bond'] = np.nanmean(np.where(dist_matrix_bond==0,np.nan,dist_matrix), axis=1)
        X['dist_median_bond'] = np.nanmedian(np.where(dist_matrix_bond==0,np.nan,dist_matrix), axis=1)
        X['dist_mean_no_bond'] = np.nanmean(np.where(dist_matrix_no_bond==0,np.nan,dist_matrix), axis=1)
        X['dist_std_no_bond'] = np.nanstd(np.where(dist_matrix_no_bond==0,np.nan,dist_matrix), axis=1)
        X['dist_median_no_bond'] = np.nanmedian(np.where(dist_matrix_no_bond==0,np.nan,dist_matrix), axis=1)
        X['dist_std'] = np.nanstd(np.where(dist_matrix==0,np.nan,dist_matrix), axis=1)
        X['dist_min'] = np.nanmin(np.where(dist_matrix==0,np.nan,dist_matrix), axis=1)
        X['dist_max'] = np.nanmax(np.where(dist_matrix==0,np.nan,dist_matrix), axis=1)
        X['range_dist'] = np.absolute(X['dist_max']-X['dist_min'])
        X['dist_bond_min'] = np.nanmin(np.where(dist_matrix_bond==0,np.nan,dist_matrix), axis=1)
        X['dist_bond_max'] = np.nanmax(np.where(dist_matrix_bond==0,np.nan,dist_matrix), axis=1)
        X['range_dist_bond'] = np.absolute(X['dist_bond_max']-X['dist_bond_min'])
        X['dist_no_bond_min'] = np.nanmin(np.where(dist_matrix_no_bond==0,np.nan,dist_matrix), axis=1)
        X['dist_no_bond_max'] = np.nanmax(np.where(dist_matrix_no_bond==0,np.nan,dist_matrix), axis=1)
        X['range_dist_no_bond'] = np.absolute(X['dist_no_bond_max']-X['dist_no_bond_min'])
        X['n_diff'] = pd.DataFrame(np.around(dist_matrix_bond,5)).nunique(axis=1).values  #5
        X = reduce_mem_usage(X,verbose=False)
        return X
    
class MakeMoreFeatures(TransformerMixin, BaseEstimator):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['distance'] = np.linalg.norm(X[['x_x','y_x','z_x']].values - X[['x_y','y_y','z_y']].values ,axis=1)
        X['x_dist'] = X['x_x'] - X['x_y']
        X['y_dist'] = X['y_x'] - X['y_y']
        X['z_dist'] = X['z_x'] - X['z_y']
        X['x_dist_abs'] = np.absolute(X['x_dist'])
        X['y_dist_abs'] = np.absolute(X['y_dist'])
        X['z_dist_abs'] = np.absolute(X['z_dist'])
        X['inv_distance3'] = 1/(X['distance']**3)
        X['dimension_x'] = np.absolute(X.groupby(['molecule_name'])['x_x'].transform('max') - X.groupby(['molecule_name'])['x_x'].transform('min'))
        X['dimension_y'] = np.absolute(X.groupby(['molecule_name'])['y_x'].transform('max') - X.groupby(['molecule_name'])['y_x'].transform('min'))
        X['dimension_z'] = np.absolute(X.groupby(['molecule_name'])['z_x'].transform('max') - X.groupby(['molecule_name'])['z_x'].transform('min'))
        X['molecule_dist_mean_x'] = X.groupby(['molecule_name'])['dist_mean_x'].transform('mean')
        X['molecule_dist_mean_y'] = X.groupby(['molecule_name'])['dist_mean_y'].transform('mean')
        X['molecule_dist_mean_bond_x'] = X.groupby(['molecule_name'])['dist_mean_bond_x'].transform('mean')
        X['molecule_dist_mean_bond_y'] = X.groupby(['molecule_name'])['dist_mean_bond_y'].transform('mean')
        X['molecule_dist_range_x'] = X.groupby(['molecule_name'])['dist_mean_x'].transform('max') - X.groupby(['molecule_name'])['dist_mean_x'].transform('min')
        X['molecule_dist_range_y'] = X.groupby(['molecule_name'])['dist_mean_y'].transform('max') - X.groupby(['molecule_name'])['dist_mean_y'].transform('min')
        X['molecule_dist_std_x'] = X.groupby(['molecule_name'])['dist_mean_x'].transform('std')
        X['molecule_dist_std_y'] = X.groupby(['molecule_name'])['dist_mean_y'].transform('std')
        X['molecule_atom_0_dist_mean'] = X.groupby(['molecule_name','atom_x'])['distance'].transform('mean')
        X['molecule_atom_1_dist_mean'] = X.groupby(['molecule_name','atom_y'])['distance'].transform('mean')
        X['molecule_atom_0_dist_std_diff'] = X.groupby(['molecule_name', 'atom_x'])['distance'].transform('std') - X['distance']
        X['molecule_atom_1_dist_std_diff'] = X.groupby(['molecule_name', 'atom_y'])['distance'].transform('std') - X['distance']
        X['molecule_type_dist_min'] = X.groupby(['molecule_name','type'])['distance'].transform('min') 
        X['molecule_type_dist_max'] = X.groupby(['molecule_name','type'])['distance'].transform('max') 
        X['molecule_dist_mean_no_bond_x'] = X.groupby(['molecule_name'])['dist_mean_no_bond_x'].transform('mean')
        X['molecule_dist_mean_no_bond_y'] = X.groupby(['molecule_name'])['dist_mean_no_bond_y'].transform('mean')
        X['molecule_atom_index_0_dist_min'] = X.groupby(['molecule_name', 'atom_index_0'])['distance'].transform('min') #new variable - dont include
        X['molecule_atom_index_0_dist_std'] = X.groupby(['molecule_name', 'atom_index_0'])['distance'].transform('std') #new variable - dont include
        X['molecule_atom_index_0_dist_min_div'] = X['molecule_atom_index_0_dist_min']/X['distance'] #new variable - include
        X['molecule_atom_index_0_dist_std_div'] = X['molecule_atom_index_0_dist_std']/X['distance'] #new variable - include
        X['molecule_atom_index_0_dist_mean'] = X.groupby(['molecule_name', 'atom_index_0'])['distance'].transform('mean') #new variable - include
        X['molecule_atom_index_0_dist_max'] = X.groupby(['molecule_name', 'atom_index_0'])['distance'].transform('max') #new variable - include
        X['molecule_atom_index_0_dist_mean_diff'] = X['molecule_atom_index_0_dist_mean'] - X['distance'] #new variable - include
        X['molecule_atom_index_1_dist_mean'] = X.groupby(['molecule_name', 'atom_index_1'])['distance'].transform('mean') #new variable - include
        X['molecule_atom_index_1_dist_max'] = X.groupby(['molecule_name', 'atom_index_1'])['distance'].transform('max') #new variable - include
        X['molecule_atom_index_1_dist_min'] = X.groupby(['molecule_name', 'atom_index_1'])['distance'].transform('min') #new variable - include
        X['molecule_atom_index_1_dist_std'] = X.groupby(['molecule_name', 'atom_index_1'])['distance'].transform('std') #new variable - dont include
        X['molecule_atom_index_1_dist_min_div'] = X['molecule_atom_index_1_dist_min']/X['distance'] #new variable - include
        X['molecule_atom_index_1_dist_std_diff'] = X['molecule_atom_index_1_dist_std'] - X['distance'] #new variable - include
        X['molecule_atom_index_1_dist_mean_div'] = X['molecule_atom_index_1_dist_mean']/X['distance'] #new variable - include
        X['molecule_atom_index_1_dist_min_diff'] = X['molecule_atom_index_1_dist_min_div'] - X['distance'] #new variable - include
        le = LabelEncoder()
        for feat in ['atom_x','atom_y']:
            le.fit(X[feat])
            X[feat] = le.transform(X[feat])
        X = reduce_mem_usage(X,verbose=False)
        return X