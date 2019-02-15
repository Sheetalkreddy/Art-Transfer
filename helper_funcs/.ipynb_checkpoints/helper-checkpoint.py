from PIL import Image
import tensorflow as tf
import numpy as np

content_layers = ['block5_conv2']

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
		 'block5_conv1'
               ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

vgg_path = './model/vgg_19/imagenet-vgg-verydeep-19.mat'


def read_image(location, width, height):
    assert isinstance(location, str), "Argument of wrong type"

    image = Image.open(location)
    tn_image = image.resize((width, height))
    
    return tn_image

def white_noise(height, width, channels):
    white_noise = np.random.random_integers(low = 0, high = 255, size=(height, width, channels)).astype("float32")
    return white_noise

