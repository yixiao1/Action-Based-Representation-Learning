import argparse

from coil_core import execute_train, execute_validation, execute_drive, folder_execute, execute_train_encoder
from coilutils.general import create_log_folder
from configs import g_conf

# You could send the module to be executed and they could have the same interface.

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        '--single-process',
        default=None,
        type=str
    )
    argparser.add_argument(
        '--gpus',
        nargs='+',
        dest='gpus',
        type=str
    )
    argparser.add_argument(
        '-nw',
        '--number-of-workers',
        dest='number_of_workers',
        type=int,
        default=12
    )
    argparser.add_argument(
        '-f',
        '--folder',
        type=str
    )
    argparser.add_argument(
        '-e',
        '--exp',
        type=str
    )
    argparser.add_argument(
        '-dk', '--docker',
        dest='docker',
        default='carlasim/carla:0.8.4',
        type=str,
        help='Set to run carla using docker'
    )
    argparser.add_argument(
        '--encoder-checkpoint',
        default=None,
        dest='encoder_checkpoint',
        type=int,
        help='The pre-trained encoder model you want to use'
    )
    argparser.add_argument(
        '-encoder-f',
        '--encoder-folder',
        default=None,
        dest='encoder_folder',
        type=str,
        help='The folder where the pre-trained encoder model is in'
    )
    argparser.add_argument(
        '-encoder-e',
        '--encoder-exp',
        default=None,
        dest='encoder_exp',
        type=str,
        help='The exp of pre-trained encoder model'
    )
    argparser.add_argument(
        '-vj', '--val-json',
        dest='val_json',
        help=' Path to the VALIDATION json file',
        default=None)

    argparser.add_argument(
        '-de',
        '--drive-envs',
        dest='driving_environments',
        nargs='+',
        default=[]
    )

    args = argparser.parse_args()

    # Check if the vector of GPUs passed are valid.
    for gpu in args.gpus:
        try:
            int(gpu)
        except ValueError:  # Reraise a meaningful error.
            raise ValueError("GPU is not a valid int number")

    # Check if the mandatory folder argument is passed
    if args.folder is None:
        raise ValueError("You should set a folder name where the experiments are placed")

    # Check if the driving parameters are passed in a correct way
    if args.driving_environments is not None:
        for de in list(args.driving_environments):
            if len(de.split('_')) < 2:
                raise ValueError("Invalid format for the driving environments should be Suite_Town")


    # This is the folder creation of the logs
    create_log_folder(args.folder)

    # The definition of parameters for driving
    drive_params = {
        "suppress_output": True,
        "docker": args.docker
    }

    # The definition of pre-trained encoder model used for training ETE
    if args.encoder_checkpoint and args.encoder_folder and args.encoder_exp:
        encoder_params = {'encoder_checkpoint': args.encoder_checkpoint,
                          'encoder_folder': args.encoder_folder,
                          'encoder_exp': args.encoder_exp}
    elif all(v is None for v in[args.encoder_checkpoint, args.encoder_folder, args.encoder_exp]):
        encoder_params = None
    else:
        print(args.encoder_folder, args.encoder_exp, args.encoder_checkpoint)
        raise ValueError("You should set all these three arugments: --encoder-folder, --encoder-exp and --encoder-checkpoint for selecting correct encoder model when training ETE by using VAE model")

    # There are two modes of execution
    if args.single_process is not None:
        ####
        # MODE 1: Single Process. Just execute a single experiment alias.
        ####

        if args.exp is None:
            raise ValueError(" You should set the exp alias when using single process")

        # train_encoder and validation_encoder are for training the encoder model only.
        if args.single_process == 'train_encoder':
            execute_train_encoder(gpu=args.gpus[0], exp_batch=args.folder, exp_alias=args.exp,
                          suppress_output=False, number_of_workers=args.number_of_workers)

        elif args.single_process == 'train':
            execute_train(gpu=args.gpus[0], exp_batch=args.folder, exp_alias=args.exp,
                          suppress_output=False, number_of_workers= args.number_of_workers,
                          encoder_params = encoder_params)

        elif args.single_process == 'validation':
            execute_validation(gpu=args.gpus[0], exp_batch=args.folder, exp_alias=args.exp,
                               json_file_path=args.val_json, suppress_output=False,
                               encoder_params = encoder_params)

        elif args.single_process == 'drive':
            # For debugging
            drive_params["suppress_output"]=False
            execute_drive(args.gpus[0], args.folder, args.exp, list(args.driving_environments)[0], drive_params, encoder_params = encoder_params)

        else:
            raise Exception("Invalid name for single process, chose from (train, validation, test)")

    else:
        """
        # TODO: Net yet implemented
        ####
        # MODE 2: Folder execution. Execute train/validation/drive for all experiments on
        #         a certain training folder
        ####
        # We set by default that each gpu has a value of 3.5, allowing a training and
        # a driving/validation
        allocation_parameters = {'gpu_value': args.gpu_value,
                                 'train_cost': 1.5,
                                 'validation_cost': 1.0,
                                 'drive_cost': 1.5}

        params = {
            'folder': args.folder,
            'gpus': list(args.gpus),
            'is_training': args.is_training,
            'validation_datasets': list(args.validation_datasets),
            'driving_environments': list(args.driving_environments),
            'driving_parameters': drive_params,
            'allocation_parameters': allocation_parameters,
            'number_of_workers': args.number_of_workers,
            'encoder_params': {'encoder_checkpoint': args.encoder_checkpoint,
                          'encoder_folder': args.encoder_folder,
                          'encoder_exp': args.encoder_exp}

        }

        folder_execute(params)
        print("SUCCESSFULLY RAN ALL EXPERIMENTS")
        
        """
        pass
