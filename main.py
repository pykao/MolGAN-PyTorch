import os
import logging

from rdkit import RDLogger

from utils.args import get_GAN_config
from utils.utils_io import get_date_postfix

# Remove flooding logs.
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from solver import Solver
from torch.backends import cudnn

def main(config):
    # For fast training
    cudnn.benchmark = True

    # Timestamp
    if config.mode == 'train':
        config.saving_dir = os.path.join(config.saving_dir, get_date_postfix())
        config.log_dir_path = os.path.join(config.saving_dir, config.mode, 'log_dir')
        config.model_dir_path = os.path.join(config.saving_dir, config.mode, 'model_dir')
        config.img_dir_path = os.path.join(config.saving_dir, config.mode, 'img_dir')
    else:
        a_test_time = get_date_postfix()
        config.saving_dir = os.path.join(config.saving_dir)
        config.log_dir_path = os.path.join(config.saving_dir, 'post_test', a_test_time, 'log_dir')
        config.model_dir_path = os.path.join(config.saving_dir, 'model_dir')
        config.img_dir_path = os.path.join(config.saving_dir, 'post_test', a_test_time, 'img_dir')


    # Create directories if not exist
    if not os.path.exists(config.log_dir_path):
        os.makedirs(config.log_dir_path)
    if not os.path.exists(config.model_dir_path):
        os.makedirs(config.model_dir_path)
    if not os.path.exists(config.img_dir_path):
        os.makedirs(config.img_dir_path)

    # Logger
    if config.mode == 'train':
        log_p_name = os.path.join(config.log_dir_path, get_date_postfix() + '_logger.log')
        logging.basicConfig(filename=log_p_name, level=logging.INFO)
        logging.info(config)

    # Solver for training and test MolGAN
    if config.mode == 'train':
        solver = Solver(config, logging)
    elif config.mode == 'test':
        solver = Solver(config)
    else:
        raise NotImplementedError

    solver.train_and_validate()

if __name__ == '__main__':

    config = get_GAN_config()

    os.environ["CUDA_VISIBLE_DEVICES"]="6"

    config.mol_data_dir = r'data/gdb9_9nodes.sparsedataset'
    #config.mol_data_dir = r'data/qm9_5k.sparsedataset'

    # Training
    config.saving_dir = r'results/GAN'
    config.z_dim = 32
    config.num_epochs = 30
    # 1.0 for pure WGAN and 0.0 for pure RL
    config.lambda_wgan = 1.0

    # Test
    #config.mode = "test"
    #config.test_epoch = 30
    #config.test_sample_size = 5000
    #config.z_dim = 32
    #config.saving_dir = r"results/GAN/20210929_175628/train"

    print(config)

    main(config)
