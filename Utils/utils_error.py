import numpy as np

from . import utils_metrics


def getErrorLabelsDict(model):
    # error_dict = {"MSE": [], "RMSE": [], "ABS": [], "PSNR": [], "SSIM": []}
    error_dict = {}
    for key in model.data_info_dict["errors_to_compute"]:
        error_dict[key] = []
    return error_dict


def computeErrors(target, prediction, data_info, single_sample=False):
    
    assert "errors_to_compute" in data_info
    errors_to_compute = data_info["errors_to_compute"]
    if single_sample:
        spatial_dims = tuple([*range(len(np.shape(target)))])
    else:
        spatial_dims = tuple([*range(len(np.shape(target)))[1:]])
    
    # ABSOLUTE ERROR
    abserror = np.abs(target - prediction)
    abserror = np.mean(abserror, axis=spatial_dims)
    
    # SQUARE ERROR
    serror = np.square(target - prediction)
    
    # MEAN (over-space) SQUARE ERROR
    mse = np.mean(serror, axis=spatial_dims)
    
    # ROOT MEAN SQUARE ERROR
    rmse = np.sqrt(mse)

    error_dict = {}

    if "CORR" in errors_to_compute:
        if single_sample:
            corr = np.corrcoef(np.reshape(target, (-1)),
                               np.reshape(prediction, (-1)))[0, 1]
        else:
            corr = np.array([np.corrcoef(np.reshape(target[t], (-1)), np.reshape(prediction[t], (-1)))[0, 1] for t in range(len(target))])
        error_dict["CORR"] = corr

    if "MSE" in errors_to_compute:
        error_dict["MSE"] = mse

    if "RMSE" in errors_to_compute:
        error_dict["RMSE"] = rmse

    if "NNAD" in errors_to_compute:
        assert "data_std" in data_info, "ERROR: data_std needed to compute the NNAD not found in the data_info_dict."
        assert "data_max" in data_info, "ERROR: data_max needed to compute the NNAD not found in the data_info_dict."
        assert "data_min" in data_info, "ERROR: data_min needed to compute the NNAD not found in the data_info_dict."
        data_std = data_info["data_std"]
        data_max = data_info["data_max"]
        data_min = data_info["data_min"]

        data_norm = data_max - data_min
        temp = len(np.shape(target))
        num_channels = temp-1 if single_sample else temp-2
        for i in range(num_channels):
            data_norm = np.expand_dims(data_norm, 1)

        if not single_sample: data_norm = data_norm[np.newaxis] # Adding the ic axis
        nrmse = np.sqrt(serror / np.power(data_norm, 2.0))
        nrmse = np.mean(nrmse, axis=spatial_dims)
        error_dict["NNAD"] = nrmse
    
    if "ABS" in errors_to_compute:
        error_dict["ABS"] = abserror

    if "NAD" in errors_to_compute:
        nad_error = np.mean(np.abs(target - prediction) /
                            (np.max(target) - np.min(target)),
                            axis=spatial_dims)
        error_dict["NAD"] = nad_error

    if "PSNR" in errors_to_compute:
        if single_sample:
            psnr = utils_metrics.PSNR(target, prediction)
        else:
            psnr = np.array([utils_metrics.PSNR(target[i], prediction[i]) for i in range(np.shape(target)[0])])
        error_dict["PSNR"] = psnr

    if "SSIM" in errors_to_compute:
        if single_sample:
            ssim = utils_metrics.SSIM(target, prediction)
        else:
            ssim = np.array([utils_metrics.SSIM(target[i], prediction[i]) for i in range(np.shape(target)[0])])
        error_dict["SSIM"] = ssim
    
    return error_dict


def getErrorDictAvg(error_dict):

    # Computing the average over time
    error_dict_avg = {}
    for key in error_dict:
        error_dict_avg[key + "_avg"] = np.mean(error_dict[key])
    printErrors(error_dict_avg)
    
    return error_dict_avg


def printErrors(error_dict):
    print("_" * 30)
    for key in error_dict:
        print("{:} = {:}".format(key, error_dict[key]))
    print("_" * 30)