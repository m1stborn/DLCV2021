import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    # basic parameters
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-4,
        help='Specify learning rate for optimizer. (default: 2e-4)')
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.5,
        help='Specify learning rate for optimizer. (default: 1e-3)')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='If set resumes training from provided checkpoint. (default: None)'
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default='./ckpt/DCGAN-cc02f2b1.pt',
        help='Path to checkpoint to resume training. (default: "")'
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default='./ckpt/p1',
        help='Path for saving checkpoint. (default: "./ckpt")'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs. (default: 20)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for data loaders. (default: 64)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of workers for data loader. (default: 8)'
    )
    parser.add_argument(
        '--test_run',
        action='store_true',
        help='Whether run the whole training data or only 1 batch per epoch. (default: False)'
    )

    # p1
    parser.add_argument(
        '--p1_train_dir',
        type=str,
        default='./hw2_data/face/train',
        help='Training Dataset dir. (default: ./hw_data/face/train)'
    )
    parser.add_argument(
        '--p1_output_base',
        type=str,
        default='./p1_result',
        help='Training Dataset dir. (default: ./p1_train_process)'
    )
    parser.add_argument(
        '--p1_output_dir',
        type=str,
        default='./p1_result/1000_generated_images',
        help='Training Dataset dir. (default: ./p1_result/1000_generated_images)'
    )
    parser.add_argument(
        '--p1_output_temp',
        type=str,
        default='./p1_result/train',
        help='Training Dataset dir. (default: ./p1_result/1000_generated_images)'
    )
    # parser.add_argument(
    #     '--p1_valid_dir',
    #     type=str,
    #     default='./p1_data/val_50',
    #     help='Valid(or Test) Dataset dir. (default: ./p1_data/val_50)'
    # )
    # parser.add_argument(
    #     '--p1_input_dir',
    #     type=str,
    #     default='./p1_data/val_50',
    #     help='Input Dataset dir. (default: ./p1_data/val_50)'
    # )
    # parser.add_argument(
    #     '--p1_output_file',
    #     type=str,
    #     default='./p1_data/output.csv',
    #     help='Input Dataset dir. (default: ./p1_data/val_50)'
    # )

    return parser
