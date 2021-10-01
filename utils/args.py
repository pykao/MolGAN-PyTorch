import argparse

def str2bool(v):
    return v.lower() in ['true']

def get_GAN_config():
    parser = argparse.ArgumentParser()

    # Model configuration
    parser.add_argument('--z_dim', type=int, default=8, help='dimension of domain labels')
    parser.add_argument('--g_conv_dim', default=[128, 256, 512], help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=[[128, 64], 128, [128, 64]], help='number of conv filters in the first layer of D')
    parser.add_argument('--lambda_wgan', type=float, default=1.0, help='weight between RL and GAN. 1.0 for Pure GAN and 0.0 for Pure RL')
    parser.add_argument('--lambda_gp', type=float, default=10.0, help='weight for gradient penalty')
    parser.add_argument('--post_method', type=str, default='softmax', choices=['softmax', 'soft_gumbel', 'hard_gumbel'])

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs for training D')
    parser.add_argument('--g_lr', type=float, default=0.001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.001, help='learning rate for D')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--resume_epoch', type=int, default=None, help='resume training from this step')

    # Test configuration
    parser.add_argument('--test_epoch', type=int, default=None, help='test model from this step')
    parser.add_argument('--test_sample_size', type=int, default=None, help='number of testing molecules')

    # Miscellaneous
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # Dataset directory
    parser.add_argument('--mol_data_dir', type=str, default='data/gdb9_9nodes.sparsedataset')

    # Saving directory
    parser.add_argument('--saving_dir', type=str, default='results/GAN/')

    # Step size
    parser.add_argument('--model_save_step', type=int, default=1)

    # Tensorboard
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    config = parser.parse_args()

    return config
