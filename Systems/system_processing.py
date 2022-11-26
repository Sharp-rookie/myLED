import os
import numpy as np

import Utils as utils

def getSystemDataInfo(model):

    # Data are of three possible data types for each application:
    # Type 1:
    #         (T, input_dim)
    # Type 2:
    #         (T, input_dim, Cx) (input_dim is the N_particles, perm_inv, etc.)
    # Type 3:
    #         (T, input_dim, Cx, Cy, Cz)

    data_info_dict = {
        'structured': False,
        'contour_plots': False,
        'density_plots': False,
        'statistics_cummulative': False,
        'statistics_per_state': False,
        'statistics_per_channel': False,  # Not implemented
        'compute_errors_in_time': False,
        'colormap': "seismic",
        # 'errors_to_compute':["MSE", "RMSE", "ABS", "PSNR", "SSIM"],
        'errors_to_compute': ["MSE", "RMSE", "NNAD", "ABS", "CORR"],
    }
    if model.params["truncate_data_batches"] > 0:
        data_info_dict.update({
            'truncate_data_batches':
            int(model.params["truncate_data_batches"])
        })

    if model.params["truncate_timesteps"] > 0:
        data_info_dict.update(
            {'truncate_timesteps': int(model.params["truncate_timesteps"])})

    assert (model.scaler in ["MinMaxZeroOne", "MinMaxMinusOneOne"])

    assert os.path.isdir(model.data_path_gen), "[getSystemDataInfo()] Data directory {:} not found.".format(model.data_path_gen)

    data_file = os.path.join(os.getcwd(), model.data_path_gen, 'data_min.txt')
    assert os.path.exists(data_file), "[getSystemDataInfo()] Data file {:} not found.".format(data_file)

    data_file = os.path.join(os.getcwd(), model.data_path_gen, 'data_max.txt')
    assert os.path.exists(
        data_file), "[getSystemDataInfo()] Data file {:} not found.".format(
            data_file)

    if model.system_name in ["Dummy"]:
        data_info_dict.update({
            'scaler':
            utils.scaler(
                scaler_type=model.scaler,
                data_min=np.array(
                    [np.loadtxt(model.data_path_gen + "/data_min.txt")]),
                data_max=np.array(
                    [np.loadtxt(model.data_path_gen + "/data_max.txt")]),
                channels=2,
                common_scaling_per_input_dim=0,
                common_scaling_per_channels=1,
            ),
            'dt':
            1,
            'statistics_cummulative':
            True,
            'compute_errors_in_time':
            True,
            'colormap':
            "seismic",
            'errors_to_compute': ["MSE", "NNAD"],
        })

    elif model.system_name in ["DummyStructured"]:
        data_info_dict.update({
            'scaler':
            utils.scaler(
                scaler_type=model.scaler,
                data_min=np.array(
                    [np.loadtxt(model.data_path_gen + "/data_min.txt")]),
                data_max=np.array(
                    [np.loadtxt(model.data_path_gen + "/data_max.txt")]),
                channels=2,
                common_scaling_per_input_dim=0,
                common_scaling_per_channels=1,
            ),
            'dt':
            1,
            'statistics_cummulative':
            True,
            'compute_errors_in_time':
            True,
            'colormap':
            "seismic",
            'structured':
            True,
        })

    elif model.system_name in ["FHN"]:

        data_info_dict.update({
            'scaler':
            utils.scaler(
                scaler_type=model.scaler,
                data_min=np.loadtxt(model.data_path_gen + "/data_min.txt"),
                data_max=np.loadtxt(model.data_path_gen + "/data_max.txt"),
                channels=1,
                common_scaling_per_input_dim=0,
                common_scaling_per_channels=1,  # Common scaling for all channels
            ),
            'dt':
            np.array([np.loadtxt(model.data_path_gen + "/dt.txt")])[0],
            'statistics_cummulative':
            True,
            'compute_errors_in_time':
            True,
            'colormap':
            "gray",
            'contour_plots':
            True,
            'errors_to_compute': ["MSE", "RMSE", "NAD", "CORR"],
        })

    elif model.system_name in ["FHNStructured"]:

        data_info_dict.update({
            'scaler':
            utils.scaler(
                scaler_type=model.scaler,
                data_min=np.loadtxt(model.data_path_gen + "/data_min.txt"),
                data_max=np.loadtxt(model.data_path_gen + "/data_max.txt"),
                channels=1,
                common_scaling_per_input_dim=0,
                common_scaling_per_channels=1,  # Common scaling for all channels
            ),
            'dt':
            np.array([np.loadtxt(model.data_path_gen + "/dt.txt")])[0],
            'statistics_cummulative':
            True,
            'compute_errors_in_time':
            True,
            'colormap':
            "gray",
            'contour_plots':
            True,
            'errors_to_compute': ["MSE", "RMSE", "NAD", "CORR"],
            'structured':
            True,
        })

    else:
        raise ValueError(
            "# Data info for system {:} not found (see system_processing.py script in folder ../Code/Methods/Codebase/Systems)."
            .format(model.system_name))
    
    return data_info_dict