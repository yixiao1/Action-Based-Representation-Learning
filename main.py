import argparse

from coil_core import execute_train, execute_validation, execute_train_encoder
from coilutils.general import create_log_folder

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
        '--encoder-checkpoint',
        default=None,
        dest='encoder_checkpoint',
        type=int,
        help='The pre-trained encoder model you want to use'
    )
    argparser.add_argument(
        '-vj', '--val-json',
        dest='val_json',
        help=' Path to the VALIDATION json file',
        default=None)

    args = argparser.parse_args()

    # Check if the vector of GPUs passed are valid.
    for gpu in args.gpus:
        try:
            int(gpu)
        except ValueError:  # Reraise a meaningful error.
            raise ValueError("GPU is not a valid int number")

    # There are two modes of execution
    if args.single_process is not None:
        if args.single_process in ['train', 'validation']:
            # Check if the mandatory folder argument is passed
            if args.folder is None:
                raise ValueError("You should set a folder name where the experiments are placed")
            # This is the folder creation of the logs
            create_log_folder(args.folder)
            if args.exp is None:
                raise ValueError("You should set the exp alias")
            # The definition of pre-trained encoder model used for training affordances
            if args.encoder_checkpoint and args.encoder_folder and args.encoder_exp:
                encoder_params = {'encoder_checkpoint': args.encoder_checkpoint,
                                  'encoder_folder': args.encoder_folder,
                                  'encoder_exp': args.encoder_exp}
            elif all(v is None for v in [args.encoder_checkpoint, args.encoder_folder, args.encoder_exp]):
                encoder_params = None
            else:
                print(args.encoder_folder, args.encoder_exp, args.encoder_checkpoint)
                raise ValueError(
                    "You should set all three arugments for using encoder: --encoder-folder, --encoder-exp and --encoder-checkpoint")

            if args.single_process == 'train':
                execute_train(gpu=args.gpus[0], exp_batch=args.folder, exp_alias=args.exp,
                              suppress_output=False, encoder_params=encoder_params)
            elif args.single_process == 'validation':
                execute_validation(gpu=args.gpus[0], exp_batch=args.folder, exp_alias=args.exp,
                                   json_file_path=args.val_json, suppress_output=False, encoder_params=encoder_params)


        # train_encoder and validation_encoder are for training the encoder model only.
        elif args.single_process == 'train_encoder':
            # Check if the mandatory folder argument is passed
            if args.encoder_folder is None:
                raise ValueError("You should set a folder name where the experiments are placed")
            # This is the folder creation of the logs
            create_log_folder(args.encoder_folder)
            if args.encoder_exp is None:
                raise ValueError("You should set the exp alias")
            execute_train_encoder(gpu=args.gpus[0], exp_batch=args.encoder_folder, exp_alias=args.encoder_exp,
                          suppress_output=False)

        else:
            raise Exception("Invalid name for single process, chose from (train, validation, test)")

    else:
        raise Exception("You need to define the process type with argument '--single-process': train_encoder, train, validation")
