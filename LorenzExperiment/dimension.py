# -*- coding: utf-8 -*-
import numpy as np
import warnings;warnings.simplefilter('ignore')

from Data.generator import generate_original_data


def cal_data_dim():

    data = generate_original_data(1, total_t=0.005, dt=0.000001, save=False)[0]
    
    # calculae ID
    def cal_id_embedding(method='MLE', is_print=False):
        from util.intrinsic_dimension import ID_Estimator
        estimator = ID_Estimator(method=method)
        k_list = (data.shape[0] * np.linspace(0.05, 0.10, 5)).astype('int')
        if is_print: print(f'[{method}] List of numbers of nearest neighbors: {k_list}')
        dims = estimator.fit(data, k_list)
        return np.mean(dims)
    
    LB_id = cal_id_embedding('MLE')
    MiND_id = cal_id_embedding('MiND_ML')
    MADA_id = cal_id_embedding('MADA')
    PCA_id = cal_id_embedding('PCA')
    MOM_id = cal_id_embedding('MOM')
    # ESS_id = cal_id_embedding('ESS')
    # DANCo_id = cal_id_embedding('DANCo')
    # TLE_id = cal_id_embedding('TLE')
    CorrInt_id = cal_id_embedding('CorrInt')

    print(f'MLE={LB_id:.1f}, MinD={MiND_id:.1f}, MADA={MADA_id:.1f}, PCA={PCA_id:.1f}, MOM={MOM_id:.1f}, CorrInt={CorrInt_id:.1f}')

cal_data_dim()