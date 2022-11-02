import warnings
import numpy as np

import Utils
import Systems


def testModesOnSet(model, set_="train", print_=False, rank_str="", gpu=False, testing_modes=[]):

    print("\n\n[Data Set]: {:}".format(set_))
    
    dt = model.data_info_dict["dt"]
    if set_ == "test":
        data_path = model.data_path_test
    elif set_ == "val":
        data_path = model.data_path_val
    elif set_ == "train":
        data_path = model.data_path_train
    else:
        raise ValueError("Invalid set {:}.".format(set_))

    data_loader_test, data_set = Utils.getDataLoader(
        data_path,
        model.data_info_dict,
        batch_size=1,
        shuffle=False,
        gpu=gpu,
    )

    testingRoutine(model, data_loader_test, dt, set_, data_set, testing_modes)


def testingRoutine(model, data_loader, dt, set_, data_set, testing_modes,):
    
    for testing_mode in testing_modes:
        testOnMode(model, data_loader, dt, set_, testing_mode, data_set)


def testOnMode(model, data_loader, dt, set_, testing_mode, data_set):
    
    assert (testing_mode in model.getTestingModes())
    assert (set_ in ["train", "test", "val"])
    print("[Test Mode]: {:}".format(testing_mode))

    # Test AE
    if testing_mode in ["autoencoder_testing"]:
        if model.data_info_dict["structured"]:
            assert False, "'structured' not implemented!"
            results = testEncodeDecodeOnHDF5Structured(model, data_set, dt, set_, testing_mode)
        else:
            results = testEncodeDecodeOnHDF5(model, data_loader, dt, set_, testing_mode)
    
    # Test RNN
    elif testing_mode in ["iterative_state_forecasting", "iterative_latent_forecasting", "teacher_forcing_forecasting"]:
        if model.data_info_dict["structured"]:
            assert False, "'structured' not implemented!"
            results = testIterativeOnHDF5Structured(model, data_set, dt, set_, testing_mode)
        else:
            results = testIterativeOnHDF5(model, data_loader, dt, set_, testing_mode)

    data_path = Utils.getResultsDir(model) + "/results_{:}_{:}".format(testing_mode, set_)
    Utils.saveData(results, data_path, model.save_format)


def testEncodeDecodeOnHDF5(model, data_loader, dt, set_name, testing_mode, dataset=None):
    
    if model.num_test_ICS > len(data_loader):
        num_test_ICS = len(data_loader)
    else:
        num_test_ICS = model.num_test_ICS

    assert num_test_ICS > 0
    print("[EncodeDecode on HDF5]: {:}/{:} initial conditions.".format(num_test_ICS, len(data_loader)))

    latent_states_all = []
    outputs_all = []
    inputs_all = []
    num_seqs_tested_on = 0

    error_dict = Utils.getErrorLabelsDict(model) # get all kind of error function name
    for input_sequence_ in data_loader:

        if num_seqs_tested_on >= num_test_ICS: break

        assert np.shape(input_sequence_)[0] == 1

        if model.data_info_dict["structured"]:
            input_sequence = dataset.getSequencesPart(input_sequence_, 0, model.prediction_horizon)
            input_sequence = input_sequence[0]
        else:
            input_sequence = input_sequence_[0]

            if model.prediction_horizon <= 0:
                raise ValueError("Prediction horizon cannot be {:}.".format(model.prediction_horizon))
            input_sequence = input_sequence[:model.prediction_horizon]

        if model.prediction_horizon > np.shape(input_sequence)[0]:
            warnings.warn("Warning: model.prediction_horizon={:} is bigger than the length of the sequence {:}.".format(model.prediction_horizon, np.shape(input_sequence)[0]))

        input_sequence = input_sequence[np.newaxis, :]

        outputs, latent_states = model.encodeDecode(input_sequence)

        input_sequence = input_sequence[0]
        latent_states = latent_states[0]
        outputs = outputs[0]

        input_sequence = model.data_info_dict["scaler"].descaleData(input_sequence, single_sequence=True, check_bounds=False, verbose=False)
        outputs = model.data_info_dict["scaler"].descaleData(outputs, single_sequence=True, check_bounds=False, verbose=False)

        errors = Utils.computeErrors(input_sequence, outputs,model.data_info_dict)
        # Updating the error
        for error in errors:
            error_dict[error].append(errors[error])

        latent_states_all.append(latent_states)
        outputs_all.append(outputs)
        inputs_all.append(input_sequence)
        num_seqs_tested_on += 1

    inputs_all = np.array(inputs_all)
    outputs_all = np.array(outputs_all)

    error_dict_avg = Utils.getErrorDictAvg(error_dict)
    results = Utils.addResultsAutoencoder(model, outputs_all, inputs_all, latent_states_all, dt, error_dict_avg)
    results = Systems.addResultsSystem(model, results, testing_mode)
    results = Systems.computeStateDistributionStatisticsSystem(model, results, inputs_all, outputs_all)
    
    return results


def testIterativeOnHDF5(model, data_loader, dt, set_name, testing_mode):

    assert (testing_mode in model.getTestingModes())

    num_test_ICS = model.num_test_ICS

    predictions_all = []
    targets_all = []
    latent_states_all = []

    predictions_augmented_all = []
    targets_augmented_all = []
    latent_states_augmented_all = []

    time_total_per_iter_all = []

    error_dict = Utils.getErrorLabelsDict(model) # get all kind of error function name

    num_max_ICS = len(data_loader)
    if num_test_ICS > num_max_ICS:
        warnings.warn("Not enough ({:}) ICs in the dataset {:}. Using {:} possible ICs.".format(num_test_ICS, set_name, num_max_ICS))
        num_test_ICS = len(data_loader)

    print("[RNN predict on HDF5]: {:}/{:} initial conditions.".format(num_test_ICS, len(data_loader)))

    ic_num = 1
    ic_indexes = []

    for sequence in data_loader:
        if ic_num > num_test_ICS: break
        if model.params["display_output"]:
            print("IC {:}/{:}, {:2.3f}%".format(ic_num, num_test_ICS, ic_num / num_test_ICS * 100))
        sequence = sequence[0]

        # STARTING TO PREDICT THE SEQUENCE IN model.predict_on=model.sequence_length
        # Warming-up with sequence_length
        model.predict_on = model.n_warmup
        assert (model.predict_on - model.n_warmup >= 0)

        if model.predict_on + model.prediction_horizon > np.shape(sequence)[0]:
            prediction_horizon = np.shape(sequence)[0] - model.predict_on
            warnings.warn("model.predict_on ({:}) + model.prediction_horizon ({:}) > np.shape(sequence)[0] ({:}). Not enough timesteps in the {:} data. Using a prediction horizon of {:}.".format(model.predict_on, model.prediction_horizon, np.shape(sequence)[0], set_name, prediction_horizon))
        else:
            prediction_horizon = model.prediction_horizon

        sequence = sequence[model.predict_on - model.n_warmup:model.predict_on + prediction_horizon]

        prediction, target, prediction_augment, target_augment, latent_states, latent_states_augmented, time_total_per_iter = model.predictSequence(
            sequence,
            testing_mode,
            dt=dt,
            prediction_horizon=prediction_horizon)

        prediction = model.data_info_dict["scaler"].descaleData(prediction, single_sequence=True, check_bounds=False)
        target = model.data_info_dict["scaler"].descaleData(target, single_sequence=True, check_bounds=False)

        prediction_augment = model.data_info_dict["scaler"].descaleData(prediction_augment, single_sequence=True, check_bounds=False)
        target_augment = model.data_info_dict["scaler"].descaleData(target_augment, single_sequence=True, check_bounds=False)

        errors = Utils.computeErrors(target, prediction,model.data_info_dict)
        
        # Updating the error
        for error in errors:
            error_dict[error].append(errors[error])

        latent_states_all.append(latent_states)
        predictions_all.append(prediction)
        targets_all.append(target)

        latent_states_augmented_all.append(latent_states_augmented)
        predictions_augmented_all.append(prediction_augment)
        targets_augmented_all.append(target_augment)

        time_total_per_iter_all.append(time_total_per_iter)
        ic_indexes.append(ic_num)
        ic_num += 1

    time_total_per_iter_all = np.array(time_total_per_iter_all)
    time_total_per_iter = np.mean(time_total_per_iter_all)

    predictions_all = np.array(predictions_all)
    targets_all = np.array(targets_all)
    latent_states_all = np.array(latent_states_all)

    predictions_augmented_all = np.array(predictions_augmented_all)
    targets_augmented_all = np.array(targets_augmented_all)
    latent_states_augmented_all = np.array(latent_states_augmented_all)

    print("Shape of trajectories:")
    print("{:}:".format(np.shape(targets_all)))
    print("{:}:".format(np.shape(predictions_all)))

    error_dict_avg = Utils.getErrorDictAvg(error_dict)

    results = Utils.addResultsIterative(
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
    )
    results = Systems.addResultsSystem(model, results, testing_mode)
    results = Systems.computeStateDistributionStatisticsSystem(model, results, targets_all, predictions_all)
    
    return results