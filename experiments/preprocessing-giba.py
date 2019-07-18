import pandas as pd
import numpy as np

from typing import *

import argparse as arg


INPUT_FEATURES: List[str] = [
 'x_0',
 'y_0', 'z_0', 'x_1', 'y_1', 'z_1', 'c_x', 'c_y', 'c_z', 'x_closest_0', 'y_closest_0',
 'z_closest_0', 'x_closest_1', 'y_closest_1', 'z_closest_1', 'distance', 'distance_center0',
 'distance_center1',  'distance_c0', 'distance_c1', 'distance_f0', 'distance_f1', 'cos_c0_c1',
 'cos_f0_f1', 'cos_center0_center1', 'cos_c0', 'cos_c1', 'cos_f0', 'cos_f1', 'cos_center0',
 'cos_center1', 'atom_n', 'adC1', 'structure_z_1', 'linkM0', 'coulomb_C.x', 'inv_dist1', 'structure_x_0',
 'inv_dist1R', 'dist_xyz', 'structure_x_1', 'distN1', 'N1', 'E0', 'coulomb_N.y', 'structure_y_1',
 'E1', 'adC3', 'distC0', 'yukawa_C.x', 'coulomb_O.y', 'vander_N.y', 'adH4', 'structure_y_0',
 'coulomb_O.x', 'yukawa_H.x', 'link1', 'min_molecule_atom_0_dist_xyz', 'distH1', 'sd_molecule_atom_0_dist_xyz',
 'yukawa_O.x', 'inv_distPE', 'inv_dist0E', 'link0', 'adN2', 'yukawa_H.y', 'NN', 'mean_molecule_atom_1_dist_xyz',
 'yukawa_N.y', 'vander_C.y', 'adN3', 'mean_molecule_atom_0_dist_xyz', 'yukawa_O.y', 'inv_dist1E',
 'max_molecule_atom_1_dist_xyz', 'linkM1', 'yukawa_F.x', 'NH', 'coulomb_N.x', 'inv_dist0', 'NO',
 'vander_F.y', 'adN1', 'structure_z_0', 'adC4', 'vander_H.x', 'R0', 'atom_index_1.1', 'NF', 'vander_C.x',
 'NC', 'ID', 'yukawa_F.y', 'distC1', 'adN4', 'pos', 'linkN', 'adH2', 'R1', 'adH3', 'typei', 'coulomb_H.x',
 'vander_O.y', 'yukawa_C.y', 'adC2', 'vander_H.y', 'coulomb_H.y', 'adH1', 'coulomb_F.y', 'max_molecule_atom_0_dist_xyz',
 'inv_distP', 'vander_F.x', 'coulomb_C.y', 'N2', 'distH0', 'yukawa_N.x', 'distN0', 'coulomb_F.x', 'vander_N.x',
 'min_molecule_atom_1_dist_xyz', 'inv_distPR', 'inv_dist0R', 'vander_O.x', 'sd_molecule_atom_1_dist_xyz']


def read_data(folder: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train = pd.read_csv(f'{folder}/train.csv')
    df_test = pd.read_csv(f'{folder}/test.csv')

    df_giba_train = pd.read_csv(f'{folder}/train_giba.csv.gz')
    df_giba_test = pd.read_csv(f'{folder}/test_giba.csv.gz')

    giba_columns_train = list(set(df_giba_train.columns).difference(set(df_train.columns)))
    giba_columns_test = list(set(df_giba_test.columns).difference(set(df_test.columns)))

    df_train = pd.concat((df_train, df_giba_train[giba_columns_train]), axis=1)
    df_test = pd.concat((df_test, df_giba_test[giba_columns_test]), axis=1)

    df_struct = pd.read_csv(f'{folder}/structures.csv')

    df_train_sub_charge = pd.read_csv(f'{folder}/mulliken_charges.csv')
    df_train_sub_tensor = pd.read_csv(f'{folder}/magnetic_shielding_tensors.csv')
                               
    return df_train, df_test, df_struct, df_train_sub_charge, df_train_sub_tensor


def reduce_mem_usage(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
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
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def map_atom_info(df_1: pd.DataFrame, df_2: pd.DataFrame, atom_idx: int) -> pd.DataFrame:
    print('Mapping...', df_1.shape, df_2.shape, atom_idx)

    df = pd.merge(df_1, df_2.drop_duplicates(subset=['molecule_name', 'atom_index']), how='left',
                  left_on=['molecule_name', f'atom_index_{atom_idx}'],
                  right_on=['molecule_name',  'atom_index'])

    df = df.drop('atom_index', axis=1)

    return df
                               
       
def complete_atom_mapping(
    df_train: pd.DataFrame, df_test: pd.DataFrame, df_struct: pd.DataFrame,
    df_train_sub_charge: pd.DataFrame, df_train_sub_tensor: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                               
    for atom_idx in [0, 1]:
        df_train = map_atom_info(df_train, df_struct, atom_idx)
        df_train = map_atom_info(df_train, df_train_sub_charge, atom_idx)
        df_train = map_atom_info(df_train, df_train_sub_tensor, atom_idx)
        df_train = df_train.rename(columns={'atom': f'atom_{atom_idx}',
                                            'x': f'x_{atom_idx}',
                                            'y': f'y_{atom_idx}',
                                            'z': f'z_{atom_idx}',
                                            'mulliken_charge': f'charge_{atom_idx}',
                                            'XX': f'XX_{atom_idx}',
                                            'YX': f'YX_{atom_idx}',
                                            'ZX': f'ZX_{atom_idx}',
                                            'XY': f'XY_{atom_idx}',
                                            'YY': f'YY_{atom_idx}',
                                            'ZY': f'ZY_{atom_idx}',
                                            'XZ': f'XZ_{atom_idx}',
                                            'YZ': f'YZ_{atom_idx}',
                                            'ZZ': f'ZZ_{atom_idx}', })
        df_test = map_atom_info(df_test, df_struct, atom_idx)
        df_test = df_test.rename(columns={'atom': f'atom_{atom_idx}',
                                          'x': f'x_{atom_idx}',
                                          'y': f'y_{atom_idx}',
                                          'z': f'z_{atom_idx}'})
        # add some features

        df_struct['c_x'] = df_struct.groupby('molecule_name')[
            'x'].transform('mean')
        df_struct['c_y'] = df_struct.groupby('molecule_name')[
            'y'].transform('mean')
        df_struct['c_z'] = df_struct.groupby('molecule_name')[
            'z'].transform('mean')
        df_struct['atom_n'] = df_struct.groupby(
            'molecule_name')['atom_index'].transform('max')

    return df_train, df_test, df_struct
       

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df['dx'] = df['x_1']-df['x_0']
    df['dy'] = df['y_1']-df['y_0']
    df['dz'] = df['z_1']-df['z_0']
    df['distance'] = (df['dx']**2+df['dy']**2+df['dz']**2)**(1/2)
    return df
                               
                               
def get_dist(df: pd.DataFrame) -> pd.DataFrame:
    df_temp = df.loc[:, ["molecule_name", "atom_index_0", "atom_index_1",
                         "distance", "x_0", "y_0", "z_0", "x_1", "y_1", "z_1"]].copy()
    df_temp_ = df_temp.copy()
    df_temp_ = df_temp_.rename(columns={'atom_index_0': 'atom_index_1',
                                        'atom_index_1': 'atom_index_0',
                                        'x_0': 'x_1',
                                        'y_0': 'y_1',
                                        'z_0': 'z_1',
                                        'x_1': 'x_0',
                                        'y_1': 'y_0',
                                        'z_1': 'z_0'})
    df_temp_all = pd.concat((df_temp, df_temp_), axis=0)

    df_temp_all["min_distance"] = df_temp_all.groupby(
        ['molecule_name', 'atom_index_0'])['distance'].transform('min')
    df_temp_all["max_distance"] = df_temp_all.groupby(
        ['molecule_name', 'atom_index_0'])['distance'].transform('max')

    df_temp = df_temp_all[df_temp_all["min_distance"]
                          == df_temp_all["distance"]].copy()
    df_temp = df_temp.drop(['x_0', 'y_0', 'z_0', 'min_distance'], axis=1)
    df_temp = df_temp.rename(columns={'atom_index_0': 'atom_index',
                                      'atom_index_1': 'atom_index_closest',
                                      'distance': 'distance_closest',
                                      'x_1': 'x_closest',
                                      'y_1': 'y_closest',
                                      'z_1': 'z_closest'})

    for atom_idx in [0, 1]:
        df = map_atom_info(df, df_temp, atom_idx)
        df = df.rename(columns={'atom_index_closest': f'atom_index_closest_{atom_idx}',
                                'distance_closest': f'distance_closest_{atom_idx}',
                                'x_closest': f'x_closest_{atom_idx}',
                                'y_closest': f'y_closest_{atom_idx}',
                                'z_closest': f'z_closest_{atom_idx}'})

    df_temp = df_temp_all[df_temp_all["max_distance"]
                          == df_temp_all["distance"]].copy()
    df_temp = df_temp.drop(['x_0', 'y_0', 'z_0', 'max_distance'], axis=1)
    df_temp = df_temp.rename(columns={'atom_index_0': 'atom_index',
                                      'atom_index_1': 'atom_index_farthest',
                                      'distance': 'distance_farthest',
                                      'x_1': 'x_farthest',
                                      'y_1': 'y_farthest',
                                      'z_1': 'z_farthest'})

    for atom_idx in [0, 1]:
        df = map_atom_info(df, df_temp, atom_idx)
        df = df.rename(columns={'atom_index_farthest': f'atom_index_farthest_{atom_idx}',
                                'distance_farthest': f'distance_farthest_{atom_idx}',
                                'x_farthest': f'x_farthest_{atom_idx}',
                                'y_farthest': f'y_farthest_{atom_idx}',
                                'z_farthest': f'z_farthest_{atom_idx}'})
    return df
                               
                               
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["distance_center0"] = (
        (df['x_0']-df['c_x'])**2+(df['y_0']-df['c_y'])**2+(df['z_0']-df['c_z'])**2)**(1/2)
    df["distance_center1"] = (
        (df['x_1']-df['c_x'])**2+(df['y_1']-df['c_y'])**2+(df['z_1']-df['c_z'])**2)**(1/2)

    df["distance_c0"] = ((df['x_0']-df['x_closest_0'])**2+(df['y_0'] -
                                                           df['y_closest_0'])**2+(df['z_0']-df['z_closest_0'])**2)**(1/2)
    df["distance_c1"] = ((df['x_1']-df['x_closest_1'])**2+(df['y_1'] -
                                                           df['y_closest_1'])**2+(df['z_1']-df['z_closest_1'])**2)**(1/2)

    df["distance_f0"] = ((df['x_0']-df['x_farthest_0'])**2+(df['y_0'] -
                                                            df['y_farthest_0'])**2+(df['z_0']-df['z_farthest_0'])**2)**(1/2)
    df["distance_f1"] = ((df['x_1']-df['x_farthest_1'])**2+(df['y_1'] -
                                                            df['y_farthest_1'])**2+(df['z_1']-df['z_farthest_1'])**2)**(1/2)

    df["vec_center0_x"] = (df['x_0']-df['c_x'])/(df["distance_center0"]+1e-10)
    df["vec_center0_y"] = (df['y_0']-df['c_y'])/(df["distance_center0"]+1e-10)
    df["vec_center0_z"] = (df['z_0']-df['c_z'])/(df["distance_center0"]+1e-10)

    df["vec_center1_x"] = (df['x_1']-df['c_x'])/(df["distance_center1"]+1e-10)
    df["vec_center1_y"] = (df['y_1']-df['c_y'])/(df["distance_center1"]+1e-10)
    df["vec_center1_z"] = (df['z_1']-df['c_z'])/(df["distance_center1"]+1e-10)

    df["vec_c0_x"] = (df['x_0']-df['x_closest_0'])/(df["distance_c0"]+1e-10)
    df["vec_c0_y"] = (df['y_0']-df['y_closest_0'])/(df["distance_c0"]+1e-10)
    df["vec_c0_z"] = (df['z_0']-df['z_closest_0'])/(df["distance_c0"]+1e-10)

    df["vec_c1_x"] = (df['x_1']-df['x_closest_1'])/(df["distance_c1"]+1e-10)
    df["vec_c1_y"] = (df['y_1']-df['y_closest_1'])/(df["distance_c1"]+1e-10)
    df["vec_c1_z"] = (df['z_1']-df['z_closest_1'])/(df["distance_c1"]+1e-10)

    df["vec_f0_x"] = (df['x_0']-df['x_farthest_0'])/(df["distance_f0"]+1e-10)
    df["vec_f0_y"] = (df['y_0']-df['y_farthest_0'])/(df["distance_f0"]+1e-10)
    df["vec_f0_z"] = (df['z_0']-df['z_farthest_0'])/(df["distance_f0"]+1e-10)

    df["vec_f1_x"] = (df['x_1']-df['x_farthest_1'])/(df["distance_f1"]+1e-10)
    df["vec_f1_y"] = (df['y_1']-df['y_farthest_1'])/(df["distance_f1"]+1e-10)
    df["vec_f1_z"] = (df['z_1']-df['z_farthest_1'])/(df["distance_f1"]+1e-10)

    df["vec_x"] = (df['x_1']-df['x_0'])/df["distance"]
    df["vec_y"] = (df['y_1']-df['y_0'])/df["distance"]
    df["vec_z"] = (df['z_1']-df['z_0'])/df["distance"]

    df["cos_c0_c1"] = df["vec_c0_x"]*df["vec_c1_x"] + \
        df["vec_c0_y"]*df["vec_c1_y"]+df["vec_c0_z"]*df["vec_c1_z"]
    df["cos_f0_f1"] = df["vec_f0_x"]*df["vec_f1_x"] + \
        df["vec_f0_y"]*df["vec_f1_y"]+df["vec_f0_z"]*df["vec_f1_z"]

    df["cos_center0_center1"] = df["vec_center0_x"]*df["vec_center1_x"] + \
        df["vec_center0_y"]*df["vec_center1_y"] + \
        df["vec_center0_z"]*df["vec_center1_z"]

    df["cos_c0"] = df["vec_c0_x"]*df["vec_x"] + \
        df["vec_c0_y"]*df["vec_y"]+df["vec_c0_z"]*df["vec_z"]
    df["cos_c1"] = df["vec_c1_x"]*df["vec_x"] + \
        df["vec_c1_y"]*df["vec_y"]+df["vec_c1_z"]*df["vec_z"]

    df["cos_f0"] = df["vec_f0_x"]*df["vec_x"] + \
        df["vec_f0_y"]*df["vec_y"]+df["vec_f0_z"]*df["vec_z"]
    df["cos_f1"] = df["vec_f1_x"]*df["vec_x"] + \
        df["vec_f1_y"]*df["vec_y"]+df["vec_f1_z"]*df["vec_z"]

    df["cos_center0"] = df["vec_center0_x"]*df["vec_x"] + \
        df["vec_center0_y"]*df["vec_y"]+df["vec_center0_z"]*df["vec_z"]
    df["cos_center1"] = df["vec_center1_x"]*df["vec_x"] + \
        df["vec_center1_y"]*df["vec_y"]+df["vec_center1_z"]*df["vec_z"]

    df = df.drop(['vec_c0_x', 'vec_c0_y', 'vec_c0_z', 'vec_c1_x', 'vec_c1_y', 'vec_c1_z',
                  'vec_f0_x', 'vec_f0_y', 'vec_f0_z', 'vec_f1_x', 'vec_f1_y', 'vec_f1_z',
                  'vec_center0_x', 'vec_center0_y', 'vec_center0_z', 'vec_center1_x',
                  'vec_center1_y', 'vec_center1_z',
                  'vec_x', 'vec_y', 'vec_z'], axis=1)
    return df

                               
if __name__ == "__main__":
    parser = arg.ArgumentParser(description='Preprocessing with additional Giba Features and 4 target output')
    parser.add_argument("--raw-folder", nargs='?', default="../data",
                        help="Folder with raw training data, without closing /")
    parser.add_argument('--output-folder', nargs='?', default="../data/champs+giba",
                        help='Folder to store processed data')

    parser.add_argument('--compress', '-c', action='store_true', help='Compress output files into .gz format')

    args = parser.parse_args()

    print(f"1 - Reading files from {args.raw_folder}")
    train, test, struct, sub_charge, sub_tensor = read_data(args.raw_folder)
    print(f"1 - Done!")

    print("2 - Mapping Data into Master Data frame")
    train, test, struct = complete_atom_mapping(train, test, struct, sub_charge, sub_tensor)
    print("2 - Done")

    print("3 - Developing Distance Features")

    print("    3.1 Trivial XYZ Distance")
    train = make_features(train)
    test = make_features(test)
    print("    3.1 Done")

    print("    3.2 Closest-Farthest features")
    train = get_dist(train)
    test = get_dist(test)
    print("    3.2 Done")

    print("    3.3 Cosine Distance Features")
    train = add_features(train)
    test = add_features(test)
    print("    3.3 Done")

    print("3 - Done")

    print(f"Train Data has {train.shape[0]} examples with {train.shape[1]} features each(including target values)")
    print(f"Test data has {test.shape[0]} examples with {test.shape[1]} features")

    print(f"4 - Saving results into {args.output_folder}")
    folder: str = args.output_folder
    if args.compress:
        file_format: str = ".hdf.gz"
    else:
        file_format: str = ".hdf"

    print("    4.1 Train features")
    train.loc[:, INPUT_FEATURES].to_hdf(f"{folder}/train_features{file_format}", "df")
    print("    4.1 Done")

    print("    4.2 Test features")
    test.loc[:, INPUT_FEATURES].to_hdf(f"{folder}/test_features{file_format}", 'df')
    print("    4.2 Done")

    print("    4.3 1-2-3 Additional Targets")
    train.loc[:, ["charge_0", "charge_1"]].to_hdf(f"{folder}/train_target_1{file_format}", 'df')
    train.loc[:, ["XX_0", "YY_0", "ZZ_0", "XX_1", "YY_1", "ZZ_1"]].to_hdf(f"{folder}/train_target_2{file_format}", 'df')
    train.loc[:, ["YX_0", "ZX_0", "XY_0", "ZY_0", "XZ_0", "YZ_0", "YX_1", "ZX_1", "XY_1", "ZY_1", "XZ_1", "YZ_1"]]\
        .to_hdf(f"{folder}/train_target_3{file_format}", 'df')
    print("    4.3 Done")

    print("    4.4 Main train target")
    train.loc[:, 'scalar_coupling_constant'].to_hdf(f"{folder}/train_label{file_format}", 'df')
    print("    4.4 Done")

    print("4 - Done")
