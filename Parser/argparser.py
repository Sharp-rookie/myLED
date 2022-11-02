import argparse
from . import parser_system
from . import parser_testing
from . import parser_crnn


def defineParser():

    parser = argparse.ArgumentParser()
    parser = parser_system.getParserSystem(parser)
    parser = parser_testing.getParserTesting(parser)
    parser = parser_crnn.getCRNNParser(parser)

    # name
    parser.add_argument("--model_name",
                        type=str,
                        required=False,
                        default="DefaultName")
    parser.add_argument("--AE_name",
                        type=str,
                        required=False,
                        default="AE")
    
    ##################################################################
    # Multiscale parameters
    ##################################################################
    parser.add_argument("--multiscale_testing",
                        help="Whether to perform the multiscale testing.",
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument("--multiscale_macro_steps_list",
                        action='append',
                        help="multiscale testing, list of macro steps to perform",
                        type=int,
                        default=[],
                        required=False)
    parser.add_argument("--multiscale_micro_steps_list",
                        action='append',
                        help="multiscale testing, list of micro steps to perform",
                        type=int,
                        default=[],
                        required=False)
    parser.add_argument("--plot_multiscale_results_comparison",
                        help="plot_multiscale_results_comparison.",
                        type=int,
                        default=0,
                        required=False)

    return parser