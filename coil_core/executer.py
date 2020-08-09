import multiprocessing
from coilutils.general import create_exp_path

from . import train, validate, train_encoder


def execute_train_encoder(gpu, exp_batch, exp_alias, suppress_output=True, number_of_workers=12):
    """

    Args:
        gpu: The gpu being used for this execution.
        module_name: The module name, if it is train, drive or evaluate
        exp_alias: The experiment alias, file name, to be executed.
        path: The path were the datasets are

    Returns:

    """
    create_exp_path(exp_batch, exp_alias)
    p = multiprocessing.Process(target=train_encoder.execute,
                                args=(gpu, exp_batch, exp_alias, suppress_output, number_of_workers))
    p.start()


def execute_train(gpu, exp_batch, exp_alias, suppress_output=True, number_of_workers=12, encoder_params = None):
    """

    Args:
        gpu: The gpu being used for this execution.
        module_name: The module name, if it is train, drive or evaluate
        exp_alias: The experiment alias, file name, to be executed.
        path: The path were the datasets are

    Returns:

    """
    if encoder_params:
        create_exp_path(exp_batch, exp_alias + '_' + str(encoder_params['encoder_checkpoint']))

    else:
        create_exp_path(exp_batch, exp_alias)

    p = multiprocessing.Process(target=train.execute,
                                args=(gpu, exp_batch, exp_alias, suppress_output, number_of_workers, encoder_params))
    p.start()


def execute_validation(gpu, exp_batch, exp_alias, json_file_path, suppress_output=True, encoder_params = None):
    """

    Args:
        gpu: The gpu being used for this execution.
        module_name: The module name, if it is train, drive or evaluate
        exp_alias: The experiment alias, file name, to be executed.
        path: The path were the datasets are

    Returns:

    """
    if encoder_params:
        create_exp_path(exp_batch, exp_alias + '_' + str(encoder_params['encoder_checkpoint']))

    else:
        create_exp_path(exp_batch, exp_alias)

    # The difference between train and validation is the
    p = multiprocessing.Process(target=validate.execute,
                                args=(gpu, exp_batch, exp_alias, json_file_path, suppress_output, encoder_params))
    p.start()