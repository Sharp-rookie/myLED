import io
import os
import torch
import psutil
import warnings
import numpy as np

from . import utils_data
from . import utils_error



def getChannels(channels, params):
    
    if channels == 0:
        Dz, Dy, Dx = 0, 0, 1
    elif channels == 1:
        assert (params["Dx"] > 0)
        Dz, Dy, Dx = 0, 0, params["Dx"]
    elif channels == 2:
        assert (params["Dy"] > 0)
        assert (params["Dx"] > 0)
        Dz, Dy, Dx = 0, params["Dy"], params["Dx"]
    elif channels == 3:
        assert (params["Dz"] > 0)
        assert (params["Dy"] > 0)
        assert (params["Dx"] > 0)
        Dz, Dy, Dx = params["Dz"], params["Dy"], params["Dx"]
    return Dz, Dy, Dx


def makeDirectories(model):
    os.makedirs(getModelDir(model), exist_ok=True)
    os.makedirs(getFigureDir(model), exist_ok=True)
    os.makedirs(getResultsDir(model), exist_ok=True)
    os.makedirs(getLogFileDir(model), exist_ok=True)
def getModelDir(model):
    if model.model_name == 'MULTISCALE':
        return model.saving_path + model.model_dir + 'RNN'
    else:
        model_dir = model.saving_path + model.model_dir + model.model_name
    return model_dir
def getFigureDir(model, unformatted=False):
    fig_dir = model.saving_path + model.fig_dir + model.model_name
    if 'multiscale' not in model.mode:
        fig_dir += '/' + 'train'
    else:
        fig_dir += '/' + 'multiscale_test'
    return fig_dir
def getResultsDir(model, unformatted=False):
    results_dir = model.saving_path + model.results_dir + model.model_name
    return results_dir
def getLogFileDir(model, unformatted=False):
    logfile_dir = model.saving_path + model.logfile_dir + model.model_name
    return logfile_dir


def transform2Tensor(model, data):
    data = toPrecision(model, data)
    data = toTensor(model, data)
    return data
def toPrecision(model, data):
    if isinstance(data, list):
        return [toPrecision(model, element) for element in data]
    if torch.is_tensor(data):
        return data.double()
    else:
        return data.astype(np.float64)
def toTensor(model, data):
    if isinstance(data, list):
        return [toTensor(model, element) for element in data]
    if not model.gpu:
        return model.torch_dtype(data)
    else:
        if torch.is_tensor(data):
            data = data.cuda()
            return model.torch_dtype(data)
        else:
            return model.torch_dtype(data)


def secondsToTimeStr(seconds):

    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        time_str = '%d[d]:%d[h]:%d[m]:%d[s]' % (days, hours, minutes, seconds)
    elif hours > 0:
        time_str = '%d[h]:%d[m]:%d[s]' % (hours, minutes, seconds)
    elif minutes > 0:
        time_str = '%d[m]:%d[s]' % (minutes, seconds)
    else:
        time_str = '%d[s]' % (seconds, )
    
    return time_str


def getMemory():

    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / 1024 / 1024
    
    return memory


def writeToLogFile(model, logfile, data, fields_to_write):
    
    with io.open(logfile, 'a+') as f:
        f.write("model_name:" + str(model.model_name))
        for field in fields_to_write:
            if (field not in data):
                raise ValueError("Field {:} is not in data.".format(field))
            f.write(":{:}:{:}".format(field, data[field]))
        f.write("\n")


def addResultsAutoencoder(model, outputs_all, inputs_all, latent_states_all, dt, error_dict_avg, latent_states_all_data=None):

    # Computing additional errors based on all predictions (e.g. frequency spectra)
    additional_results_dict, additional_errors_dict = computeAdditionalResults(model, outputs_all, inputs_all, dt)

    if latent_states_all_data == None:
        latent_states_all_data = latent_states_all
    latent_state_info = computeLatentStateInfo(model, latent_states_all_data)

    results = {
        "dt": dt,
        "latent_states_all": latent_states_all,
        "outputs_all": outputs_all,
        "inputs_all": inputs_all,
        "latent_state_info": latent_state_info,
        "fields_2_save_2_logfile": [],
    }
    results["fields_2_save_2_logfile"] += list(error_dict_avg.keys())
    results = {
        **results,
        **error_dict_avg,
        **additional_results_dict,
        **additional_errors_dict
    }

    state_statistics = computeStateDistributionStatistics(model, inputs_all, outputs_all)

    results = {**results, **state_statistics}
    
    return results


def addResultsIterative(
    model,
    predictions_all,
    targets_all,
    latent_states_all,
    predictions_augmented_all,
    targets_augmented_all,
    latent_states_augmented_all,
    time_total_per_iter,
    testing_mode,
    ic_indexes,
    dt,
    error_dict,
    error_dict_avg,
    latent_states_all_data=None,
):

    additional_results_dict, additional_errors_dict = computeAdditionalResults(model, predictions_all, targets_all, dt)
    error_dict_avg = {**error_dict_avg, **additional_errors_dict}

    state_statistics = computeStateDistributionStatistics(model, targets_all, predictions_all)

    fields_2_save_2_logfile = ["time_total_per_iter",]
    fields_2_save_2_logfile += list(error_dict_avg.keys())

    results = {
        "fields_2_save_2_logfile": fields_2_save_2_logfile,
        "predictions_all": predictions_all,
        "targets_all": targets_all,
        "latent_states_all": latent_states_all,
        "predictions_augmented_all": predictions_augmented_all,
        "targets_augmented_all": targets_augmented_all,
        "latent_states_augmented_all": latent_states_augmented_all,
        "n_warmup": model.n_warmup,
        "testing_mode": testing_mode,
        "dt": dt,
        "time_total_per_iter": time_total_per_iter,
        "ic_indexes": ic_indexes,
    }
    results = {
        **results,
        **additional_results_dict,
        **error_dict,
        **error_dict_avg,
    }
    return results


def computeLatentStateInfo(model, latent_states_all):
    #########################################################
    # In case of plain CNN (no MLP between encoder-decoder):
    # shape either  (n_ics, T, latent_state, 1, 1)
    # shape or      (n_ics, T, 1, 1, latent_state)
    #########################################################
    # In case of CNN-MLP (encoder-MLP-latent_space-decoder):
    # shape either  (n_ics, T, latent_state)
    # Case (n_ics, T, latent_state)
    
    assert len(np.shape(latent_states_all)) == 3, "np.shape(latent_states_all)={:}".format(np.shape(latent_states_all))
    latent_states_all = np.reshape(latent_states_all, (-1, model.latent_state_dim))
    min_ = np.min(latent_states_all, axis=0)
    max_ = np.max(latent_states_all, axis=0)
    mean_ = np.mean(latent_states_all, axis=0)
    std_ = np.std(latent_states_all, axis=0)
    latent_state_info = {}
    latent_state_info["min"] = min_
    latent_state_info["max"] = max_
    latent_state_info["mean"] = mean_
    latent_state_info["std"] = std_
    
    return latent_state_info


def computeStateDistributionStatistics(model, targets_all, predictions_all):
    # Computing statistical errors on distributions.
    # In case on output states that are all the same (Kuramoto-Sivashinsky) the state distribution is calculated with respect to all states.
    # In case on outputs that are different (Alanine), the state distributions are calculated with respect to each output separately.
    
    state_dist_statistics = {}

    if model.data_info_dict["structured"]:
        warnings.warn("[computeStateDistributionStatistics()] Not implemented for structured data.")
        return state_dist_statistics

    return state_dist_statistics


def computeAdditionalResults(model, predictions_all, targets_all, dt):
    
    additional_errors_dict = {}
    additional_results_dict = {}

    if model.params["compute_spectrum"]:
        if model.data_info_dict["structured"]:
            raise ValueError("Not implemented.")

        freq_pred, freq_true, sp_true, sp_pred, error_freq = computeFrequencyError(predictions_all, targets_all, dt)
        additional_results_dict["freq_pred"] = freq_pred
        additional_results_dict["freq_true"] = freq_true
        additional_results_dict["sp_true"] = sp_true
        additional_results_dict["sp_pred"] = sp_pred
        additional_errors_dict["error_freq"] = error_freq

    if model.data_info_dict["compute_errors_in_time"]:

        error_mean_dict_in_time, error_std_dict_in_time = comptuteErrorsInTime(model, predictions_all, targets_all)
        additional_results_dict["error_mean_dict_in_time"] = error_mean_dict_in_time
        additional_results_dict["error_std_dict_in_time"] = error_std_dict_in_time

    return additional_results_dict, additional_errors_dict


def computeFrequencyError(predictions_all, targets_all, dt):
    
    spatial_dims = len(np.shape(predictions_all)[2:])
    if spatial_dims == 1:
        sp_pred, freq_pred = computeSpectrum(predictions_all, dt)
        sp_true, freq_true = computeSpectrum(targets_all, dt)
        # s_dbfs = 20 * np.log10(s_mag)
        # TRANSFORM TO AMPLITUDE FROM DB
        sp_pred = np.exp(sp_pred / 20.0)
        sp_true = np.exp(sp_true / 20.0)
        error_freq = np.mean(np.abs(sp_pred - sp_true))
        return freq_pred, freq_true, sp_true, sp_pred, error_freq
    elif spatial_dims == 3:
        # RGB Image channells (Dz) of Dx x Dy
        # Applying two dimensional FFT
        sp_true = computeSpectrum2D(targets_all)
        sp_pred = computeSpectrum2D(predictions_all)
        error_freq = np.mean(np.abs(sp_pred - sp_true))
        return None, None, sp_pred, sp_true, error_freq
    elif spatial_dims == 2:
        nics, T, n_o, Dx = np.shape(predictions_all)
        predictions_all = np.reshape(predictions_all, (nics, T, n_o * Dx))
        targets_all = np.reshape(targets_all, (nics, T, n_o * Dx))
        sp_pred, freq_pred = computeSpectrum(predictions_all, dt)
        sp_true, freq_true = computeSpectrum(targets_all, dt)
        # s_dbfs = 20 * np.log10(s_mag)
        # TRANSFORM TO AMPLITUDE FROM DB
        sp_pred = np.exp(sp_pred / 20.0)
        sp_true = np.exp(sp_true / 20.0)
        error_freq = np.mean(np.abs(sp_pred - sp_true))
        return freq_pred, freq_true, sp_true, sp_pred, error_freq
    else:
        raise ValueError("Not implemented. Shape of predictions_all is {:}, with spatial_dims={:}.".format(np.shape(predictions_all), spatial_dims))


def comptuteErrorsInTime(model, targets_all, predictions_all):
    # np.shape(targets_all) = [N_ICS, N_TIMESTEP, 1, Dx]
    # np.shape(targets_all) = [N_ICS, N_TIMESTEP, 1, Dy, Dx] OR
    # np.shape(targets_all) = [N_ICS, N_TIMESTEP, 1, Dz, Dy, Dx]

    if model.data_info_dict["structured"]:
        T = np.shape(targets_all)[1]
        N_ICS = np.shape(targets_all)[0]
        error_mean_dict_in_time = utils_error.getErrorLabelsDict(model)
        error_std_dict_in_time = utils_error.getErrorLabelsDict(model)
        for t in range(T):

            error_dict = utils_error.getErrorLabelsDict(model)
            for ic in range(N_ICS):
                target_path = targets_all[ic][t]
                prediction_path = predictions_all[ic][t]

                target = utils_data.getDataHDF5Field(target_path[0], target_path[1])
                prediction = utils_data.getDataHDF5Field(prediction_path[0], prediction_path[1])

                error_dict_t_ic = utils_error.computeErrors(target, prediction, model.data_info_dict, single_sample=True)
                for key in error_dict_t_ic:
                    error_dict[key].append(error_dict_t_ic[key])

            # Mean over initial conditions
            error_mean_dict = {}
            error_std_dict = {}

            for key in error_dict:
                error_mean_dict_in_time[key].append(np.mean(np.array(error_dict[key])))
                error_std_dict_in_time[key].append(np.std(np.array(error_dict[key])))
    else:
        targets_all = np.swapaxes(targets_all, 0, 1)
        predictions_all = np.swapaxes(predictions_all, 0, 1)

        T = np.shape(targets_all)[0]
        N_ICS = np.shape(targets_all)[1]
        input_dim = np.shape(targets_all)[2]
        error_mean_dict_in_time = utils_error.getErrorLabelsDict(model)
        error_std_dict_in_time = utils_error.getErrorLabelsDict(model)

        for t in range(T):
            target = targets_all[t]
            prediction = predictions_all[t]
            error_dict = utils_error.computeErrors(target, prediction, model.data_info_dict)
            # Mean over initial conditions
            error_mean_dict = {}
            error_std_dict = {}
            for key in error_dict:
                error_mean_dict_in_time[key].append(np.mean(np.array(error_dict[key])))
                error_std_dict_in_time[key].append(np.std(np.array(error_dict[key])))
    
    return error_mean_dict_in_time, error_std_dict_in_time


def computeSpectrum(data_all, dt):
    
    # Of the form [n_ics, T, n_dim]
    spectrum_db = []
    for data in data_all:
        data = np.transpose(data)
        for d in data:
            freq, s_dbfs = dbfft(d, 1 / dt)
            spectrum_db.append(s_dbfs)
    spectrum_db = np.array(spectrum_db).mean(axis=0)
    
    return spectrum_db, freq

def computeSpectrum2D(data_all):
    
    # Of the form [n_ics, T, n_dim]
    spectrum_db = []
    for data_ic in data_all:
        # data_ic shape = T, 1(Dz), 65(Dx), 65(Dy)
        for data_t in data_ic:
            # Taking accoung only the first channel
            s_dbfs = dbfft2D(data_t[0])
            spectrum_db.append(s_dbfs)
    # MEAN OVER ALL ICS AND ALL TIME-STEPS
    spectrum_db = np.array(spectrum_db).mean(axis=0)
    
    return spectrum_db


def dbfft(x, fs):
    # !!! TIME DOMAIN FFT !!!
    """
    Calculate spectrum in dB scale
    Args:
        x: input signal
        fs: sampling frequency
    Returns:
        freq: frequency vector
        s_db: spectrum in dB scale
    """
    
    N = len(x)  # Length of input sequence
    if N % 2 != 0:
        x = x[:-1]
        N = len(x)
    x = np.reshape(x, (1, N))
    # Calculate real FFT and frequency vector
    sp = np.fft.rfft(x)
    freq = np.arange((N / 2) + 1) / (float(N) / fs)
    # Scale the magnitude of FFT by window and factor of 2,
    # because we are using half of FFT spectrum.
    s_mag = np.abs(sp) * 2 / N
    # Convert to dBFS
    s_dbfs = 20 * np.log10(s_mag)
    s_dbfs = s_dbfs[0]
    
    return freq, s_dbfs


def dbfft2D(x):
    # !! SPATIAL FFT !!
    
    N = len(x)  # Length of input sequence
    if N % 2 != 0:
        x = x[:-1, :-1]
        N = len(x)
    x = np.reshape(x, (N, N))
    # Calculate real FFT and frequency vector
    sp = np.fft.fft2(x)
    sp = np.fft.fftshift(sp)
    s_mag = np.abs(sp)
    # Convert to dBFS
    s_dbfs = 20 * np.log10(s_mag)
    
    return s_dbfs