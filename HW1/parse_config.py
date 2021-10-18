import argparse


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Specify learning rate for optimizer. (default: 1e-3)')
    # TODO: resume training
    parser.add_argument(
        '--resume',
        action='store_true',
        help='If set resumes training from provided checkpoint. (default: None)'
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default='./ckpt/',
        help='Path to checkpoint to resume training. (default: "")'
    )
    parser.add_argument(
        '--path_to_checkpoint',
        type=str,
        default='./ckpt/',
        help='Path for saving checkpoint. (default: "./ckpt/")'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs. (default: 2)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for data loaders. (default: 4)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of workers for data loader. (default: 8)'
    )
    parser.add_argument(
        '--test_run',
        type=bool,
        default=False,
        help='Whether run the whole training data or only 1 batch per epoch. (default: False)'
    )
    return parser
