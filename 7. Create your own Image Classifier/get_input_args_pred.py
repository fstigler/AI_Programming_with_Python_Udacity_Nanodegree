import argparse

def get_input_args_pred():

# Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='uploaded_images/', help='path to the image to be predicted')

    parser.add_argument('--top_k', type=int, default=4, help='no. of top prediction classes')

    parser.add_argument('--checkpoint', type=str, default='./checkpoint.pth', help='path where checkpoint is saved')

    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='map categories to real names')

    parser.add_argument('--image_name', type=str, default='image_07097.jpg', help='name of the image to be predicted')
    
    parser.add_argument('--gpu', action='store_true', help='gpu option')
    
    return parser.parse_args()
