import argparse

def get_input_args():

    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='flowers', help='path to the folder of flowers images')

#parser.add_argument

    parser.add_argument('--arch', type=str, default='vgg16', help='CNN model for classifier')

    parser.add_argument('--save_dir', type=str, default='save_directory', help='path to save checkpoints')

    parser.add_argument('--epochs', type=int, default=3, help='epochs to train the network')

    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate of network')

    parser.add_argument('--hidden_units', type=int, default=4096, help='number of hidden units')
    
    parser.add_argument('--gpu', action='store_true', help='gpu option')
    
    return parser.parse_args()
