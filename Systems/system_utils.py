

def checkSystemName(model):
    
    isGood = False
    if model.system_name in ["FHN", "FHNStructured"]:
        isGood = True
    
    return isGood


def addResultsSystem(model, results, testing_mode):
    
    if "FHN" in model.system_name:
        from .FHN import utils_processing_fhn as utils_processing_fhn
        results = utils_processing_fhn.addResultsSystemFHN(model, results, testing_mode)

    return results


def computeStateDistributionStatisticsSystem(model, state_dist_statistics, targets_all, predictions_all):

    return state_dist_statistics
