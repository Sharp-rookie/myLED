import torch
import numpy as np

import Utils
import Systems
from . import utils_multiscale_unstructured as mutils_unstructured
from . import utils_multiscale_plotting as utils_multiscale_plotting

class multiscaleTestingClass:
    def __init__(self, model, params_dict):
        
        super(multiscaleTestingClass, self).__init__()
        self.model = model
        self.params = params_dict

        self.multiscale_micro_steps_list = params_dict["multiscale_micro_steps_list"]
        self.multiscale_macro_steps_list = params_dict["multiscale_macro_steps_list"]
        self.model_class = self.model.__class__.__name__
        assert self.model_class in ["crnn"]

        """ Adding parameters to the data_info_dict """
        self.microdynamics_info_dict = Systems.getMicrodynamicsInfo(self.model)
    

    def test(self):

        self.model.load()

        with torch.no_grad():
            if self.params["n_warmup"] is None:
                self.model.n_warmup = 0
            else:
                self.model.n_warmup = int(self.params["n_warmup"])
            
            test_on = []
            
            if self.params["test_on_test"]: test_on.append("test")
            if self.params["test_on_val"]: test_on.append("val")
            if self.params["test_on_train"]: test_on.append("train")
            
            for set_ in test_on:
                self.testOnSet(set_)
    

    def plot(self):
        self.writeLogfiles()
        if self.model.params["plotting"]:
            self.plot_()
        else:
            print("plotting=0, no plot")
    

    def testOnSet(self, set_="train"):

        print("[Data Set]: {:}".format(set_))
        
        #Macro scale dt
        dt = self.model.data_info_dict["dt"]
        print("[Macro scale] dt = {:}".format(dt))

        if set_ == "test":
            data_path = self.model.data_path_test
        elif set_ == "val":
            data_path = self.model.data_path_val
        elif set_ == "train":
            data_path = self.model.data_path_train
        else:
            raise ValueError("Invalid set {:}.".format(set_))

        data_loader_test, data_set = Utils.getDataLoader(
            data_path,
            self.model.data_info_dict,
            batch_size=1,
            shuffle=False,
        )

        self.testingRoutine(data_loader_test, dt, set_, data_set)
    

    def testingRoutine(self, data_loader, dt, set_, data_set,):

        for testing_mode in self.getMultiscaleTestingModes():
            self.testOnMode(data_loader, dt, set_, testing_mode, data_set)
    

    def getMultiscaleTestingModes(self):
        
        modes = []
        for micro_steps in self.multiscale_micro_steps_list:
            for macro_steps in self.multiscale_macro_steps_list:
                mode = "multiscale_forecasting_micro_{:}_macro_{:}".format(int(micro_steps), int(macro_steps))
                modes.append(mode)
        
        return modes
    

    def testOnMode(self, data_loader, dt, set_, testing_mode, data_set):

        assert (testing_mode in self.getMultiscaleTestingModes())
        assert (set_ in ["train", "test", "val"])
        print("\n\n[Testing Mode]: {:}".format(testing_mode))

        if self.model.num_test_ICS > 0:
            if self.model.data_info_dict["structured"]:
                assert False, "structured not implemented!"
                # Testing on structured data
                results = mutils_structured.predictIndexesOnStructured(self, data_set, dt, set_, testing_mode)

            else:
                results = mutils_unstructured.predictIndexes(self, data_loader, dt, set_, testing_mode)

            data_path = Utils.getResultsDir(self.model) + "/results_{:}_{:}".format(testing_mode, set_)
            Utils.saveData(results, data_path, self.model.save_format)
        
        else:
            print("[utils_multiscale] Model has RNN but no initial conditions set to test num_test_ICS={:}.".format(self.model.num_test_ICS))
    

    def getMultiscaleParams(self, testing_mode, prediction_horizon):
        
        temp = testing_mode.split("_")
        multiscale_macro_steps = int(float(temp[-1]))
        multiscale_micro_steps = int(float(temp[-3]))
        macro_steps_per_round = []
        micro_steps_per_round = []
        
        steps = 0
        while (steps < prediction_horizon):
            steps_to_go = prediction_horizon - steps
            if steps_to_go >= multiscale_macro_steps:
                macro_steps_per_round.append(multiscale_macro_steps)
                steps += multiscale_macro_steps
            elif steps_to_go != 0:
                macro_steps_per_round.append(steps_to_go)
                steps += steps_to_go
            else:
                raise ValueError("This was not supposed to happen.")
            steps_to_go = prediction_horizon - steps
            if steps_to_go >= multiscale_micro_steps:
                micro_steps_per_round.append(multiscale_micro_steps)
                steps += multiscale_micro_steps
            elif steps_to_go != 0:
                micro_steps_per_round.append(steps_to_go)
                steps += steps_to_go

        print("[utils_multiscale] macro_steps_per_round: \n[utils_multiscale] {:}".format(macro_steps_per_round))
        print("[utils_multiscale] micro_steps_per_round: \n[utils_multiscale] {:}".format(micro_steps_per_round))
        multiscale_rounds = np.max([len(micro_steps_per_round), len(macro_steps_per_round)])
        print("[utils_multiscale] multiscale_rounds: \n[utils_multiscale] {:}". format(multiscale_rounds))
        
        return multiscale_rounds, macro_steps_per_round, micro_steps_per_round, multiscale_micro_steps, multiscale_macro_steps
    

    def writeLogfiles(self):

        write_logs_on = []
        if self.model.params["test_on_test"]: write_logs_on.append("test")
        if self.model.params["test_on_val"]: write_logs_on.append("val")
        if self.model.params["test_on_train"]: write_logs_on.append("train")

        for set_name in write_logs_on:

            # Postprocessing of RNN testing results
            for testing_mode in self.getMultiscaleTestingModes():

                # Loading the results
                data_path = Utils.getResultsDir(self.model) + "/results_{:}_{:}".format(testing_mode, set_name)
                results = Utils.loadData(data_path, self.model.save_format)

                if self.model.write_to_log:
                    logfile = Utils.getLogFileDir(self.model) + "/results_{:}_{:}.txt".format(testing_mode, set_name)
                    Utils.writeToLogFile(self.model, logfile, results, results["fields_2_save_2_logfile"])
    

    def plot_(self):

        plot_on = []
        if self.params["test_on_test"]: plot_on.append("test")
        if self.params["test_on_val"]: plot_on.append("val")
        if self.params["test_on_train"]: plot_on.append("train")

        for set_name in plot_on:

            fields_to_compare = self.getFieldsToCompare()
            fields_to_compare = Systems.addFieldsToCompare(self.model, fields_to_compare)

            dicts_to_compare = {}
            latent_states_dict = {}

            write_logs_on = []

            for testing_mode in self.getMultiscaleTestingModes():

                # Loading the results
                data_path = Utils.getResultsDir(self.model) + "/results_{:}_{:}".format(testing_mode, set_name)
                results = Utils.loadData(data_path, self.model.save_format)

                # Plotting the state distributions specific to a system
                if self.model.params["plot_system"]:
                    Systems.plotSystem(self.model, results, set_name, testing_mode)

                if self.model.params["plot_errors_in_time"]:
                    Utils.plotErrorsInTime(self.model, results, set_name, testing_mode)

                ic_indexes = results["ic_indexes"]
                dt = results["dt"]
                n_warmup = results["n_warmup"]

                predictions_augmented_all = results["predictions_augmented_all"]
                targets_augmented_all = results["targets_augmented_all"]

                predictions_all = results["predictions_all"]
                targets_all = results["targets_all"]
                latent_states_all = results["latent_states_all"]

                latent_states_dict[testing_mode] = latent_states_all

                results_dict = {}
                for field in fields_to_compare:
                    results_dict[field] = results[field]
                dicts_to_compare[testing_mode] = results_dict

                if self.model.params["plot_testing_ics_examples"]:

                    max_index = np.min([3, np.shape(results["targets_all"])[0]])

                    for idx in range(max_index):
                        print("[utils_multiscale] Plotting IC {:}/{:}.".format(idx, max_index))

                        # Plotting the latent dynamics for these examples
                        if self.model.params["plot_latent_dynamics"]:
                            Utils.plotLatentDynamics(self.model, set_name, latent_states_all[idx], idx, testing_mode)

                        results_idx = {
                            "Reference": targets_all[idx],
                            "prediction": predictions_all[idx],
                            "latent_states": latent_states_all[idx],
                            "fields_2_save_2_logfile": [],
                        }

                        self.model.parent = self
                        Utils.createIterativePredictionPlots(self.model, \
                            targets_all[idx], \
                            predictions_all[idx], \
                            dt, idx, set_name, \
                            testing_mode=testing_mode, \
                            latent_states=latent_states_all[idx], \
                            warm_up=n_warmup, \
                            target_augment=targets_augmented_all[idx], \
                            prediction_augment=predictions_augmented_all[idx], \
                            )

            if self.model.params["plot_multiscale_results_comparison"]:
                utils_multiscale_plotting.plotMultiscaleResultsComparison(
                    self.model,
                    dicts_to_compare,
                    set_name,
                    fields_to_compare,
                    results["dt"],
                )
    

    def getFieldsToCompare(self):

        error_labels = Utils.getErrorLabelsDict(self.model)
        error_labels = error_labels.keys()
        fields_to_compare = [key for key in error_labels]

        fields_to_compare.append("time_total_per_iter")

        return fields_to_compare