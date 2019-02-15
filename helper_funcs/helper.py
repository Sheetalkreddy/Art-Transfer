import numpy as np
import math
from PIL import Image
from helper_funcs.vgg import VGG
import tensorflow as tf
from helper_funcs.helper import *
import matplotlib.pyplot as plt
import scipy.misc
from time import time

vgg_path = './model/vgg_19/imagenet-vgg-verydeep-19.mat'

# Function to read images
def read_image(location, width, height):
    assert isinstance(location, str), "Argument of wrong type"

    image = Image.open(location)
    tn_image = image.resize((width, height))
    
    return tn_image

#Function to generate white noise that will be used to generate output
def white_noise(height, width, channels, content_image):
    white_noise = np.random.random_integers(low = -30, high = 30, size=(height, width, channels)).astype("float32")
    noise = white_noise * 0.5 + content_image * 0.5
    return noise

#Calculating the content loss
def content_loss(white_noise_vgg, c_img, content_layer, vgg_path = './model/vgg_19/imagenet-vgg-verydeep-19.mat'):
    
    # Creating a VGG model with content image as input, and extracting values at each layer    
    content_vgg = VGG(data_path=vgg_path)
    vgg_input = tf.placeholder('float', shape=c_img.shape)
    c_img_norm = content_vgg.normalize(c_img)
    c_img_norm = c_img_norm.astype("float32")
    content_vgg = content_vgg.vgg_net(c_img_norm)
    
    with tf.Session() as sess:
        content_layer_mat = content_vgg[content_layer].eval(feed_dict={vgg_input: c_img_norm})
        batch , height, width, num_filters = content_layer_mat.shape
        feature_content = tf.reshape(content_layer_mat, (num_filters , height*width))
        feature_noise = tf.reshape(white_noise_vgg[content_layer], (num_filters , height*width))
    c_loss = 0.5 * tf.reduce_sum(tf.pow(feature_content - feature_noise,2))
    
    return c_loss

#Calculating the style loss
def style_loss(white_noise_vgg, s_img, style_layers, weights, vgg_path = './model/vgg_19/imagenet-vgg-verydeep-19.mat'):
    # Calculating correlation of style features
    def gram_matrix(feature):
        batch , height, width, num_filters = feature.shape
        flatten = tf.reshape(feature, (height*width , num_filters))
        matrix = tf.matmul(tf.transpose(flatten) , flatten)   
        return matrix
        
    # Creating a VGG model with style image as input, and extracting values at each layer    
    style_vgg = VGG(data_path=vgg_path)
    vgg_input = tf.placeholder('float', shape=s_img.shape)
    s_img_norm = style_vgg.normalize(s_img)
    s_img_norm = s_img_norm.astype("float32")
    style_vgg = style_vgg.vgg_net(s_img_norm)
    loss = []
    
    for j,i in enumerate(style_layers):
        with tf.Session() as sess:
            feature = style_vgg[i].eval(feed_dict={vgg_input: s_img_norm})
            batch , height, width, num_filters = feature.shape
            loss.append(weights[j]*1/(4*(num_filters**2)*(height**2)*(width**2)) * tf.reduce_sum(tf.pow(gram_matrix(feature)- gram_matrix(white_noise_vgg[i]), 2)))
    
    net_loss = sum(loss)
    return net_loss

#Training on the white noise image to generate the output image.
def train(white_noise_img,c_img, s_img, style_layers, content_layer,weights,weight_c, weight_s, vgg_path, num_epochs , learning_rate = 1, output_path = "./data/output/outfile.jpg"):
    
    # Creating a VGG model with the white noise as input and extracting values at each layer
    white_noise_vgg = VGG(data_path= vgg_path)
    white_noise_img_norm = white_noise_vgg.normalize(white_noise_img)
    white_noise_img_norm = white_noise_img_norm.astype("float32")
    white_noise_img_norm = tf.get_variable("white_noise_img", initializer = white_noise_img_norm)
    white_noise_vgg = white_noise_vgg.vgg_net(white_noise_img_norm)
    
    # extracting content and style loss
    content_loss_val = content_loss(white_noise_vgg, c_img, content_layer, vgg_path)
    style_loss_val = style_loss(white_noise_vgg, s_img, style_layers, weights, vgg_path)
    
    #Optimizer for training
    total_loss = weight_c * content_loss_val + weight_s * style_loss_val
    step = tf.train.RMSPropOptimizer(learning_rate).minimize(total_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_epochs):
            step.run()
            loss = total_loss.eval()
            image = white_noise_img_norm.eval()
            if i == 0:
                print("Epoch: " + str(i+1) + " Loss = " + str(loss))
            elif (i+1)%100 == 0:
                print("Epoch: " + str(i+1) + " Loss = " + str(loss))
        # Saving the final image        
        final_image = VGG(data_path=vgg_path).unnormalize(image)
        final_image = final_image[0,:,:,:]
        final_image = np.clip(final_image, 0, 255).astype('uint8')
        scipy.misc.imsave(output_path, final_image)



