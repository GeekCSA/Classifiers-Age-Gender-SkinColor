import matplotlib as matplotlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import os
import glob
import math
import scipy.misc as smp
from PIL import Image
import time
import random


# %matplotlib inline

heigh_global=56
width_global=56

def get_batches(folder_path, image_type, size_batch = 1, last_throw = False):

    '''
    The function groups the images into a batch, each containing 'batch_size' images

    :Args
        :param folder_path (str): The folder path of the images
        :param image_type (str): The type of the images (format: '*.png', '*.jpg' etc.)
        :param size_batch (int): The size of each batch
        :param last_throw (bool): If throw the last batch if it is not full
    :Returns
        list: List of batches, each batch holds #size_batch names (of files) and #size_batch images (pixels)
        [ [filename1.jpg,filename2.jpg,filename3.jpg],[ [R1,G1,B1],[R2,G2,B2],[R3,G3,B3] ],     [filename4.jpg,]... ]

        int: The number of images in the last batch (if it is not full, if full then the number is 0)
    :Raises
        ValueError: If the value of size_batch is less than 1
    '''

    if size_batch < 1:
        raise ValueError('The size_batch should be greater than 0. The value of size_batch was: {}'.format(size_batch))

    # num_of_images is number files in this folder
    num_of_images = len(next(os.walk(folder_path))[2])

    files_name = os.path.join(folder_path, image_type)

    # glob goes through all files in dir
    images_name = [file_path for file_path in glob.iglob(files_name)] # List of files name
    # images_pixels = mpimg.imread(images_name[0])
    # contains all the pixels of evey file in this dir
    images_pixels = [mpimg.imread(file_path) for file_path in images_name] # List of pixels # Write "mpimg.imread(file_path)[:,:,0]" for one color
    images_pixels=np.array(images_pixels)
    images_pixels=images_pixels/255.0

    batches = []

    for i in range(0, num_of_images - size_batch, size_batch):
        batch = [images_name[i: i + size_batch],images_pixels[i: i + size_batch]]
        batches.append(batch)

    no_enter_to_batch = num_of_images - len(batches)*size_batch

    if((not last_throw and no_enter_to_batch != 0) or no_enter_to_batch == size_batch):
        end_batch_names = images_name[-no_enter_to_batch:]
        end_batch_pixels = images_pixels[-no_enter_to_batch:]
        end_batch = [end_batch_names, end_batch_pixels]
        batches.append(end_batch)

        if no_enter_to_batch == size_batch:
            no_enter_to_batch = 0

    return batches, no_enter_to_batch

def extract_details_from_file_name(batches):

    '''
    The function extract the details of image from the image name (name of file).
    The details contain age, gender and skin color.

    :Args
        :param batches (list): The output from "get_batches" function
    '''

    for batch in batches:
        batch_details = []
        for name in batch[0]:
            age_gender_color_by_regex = re.search('([0-9]{1,3})_([0-1])_([0-4])', name)
            details = [0] * 7 # Create a list that contains zeros. the number 7 is for [age, gender, color1, color2, color3, color4, color5]
            details[0] = int(age_gender_color_by_regex.group(1).replace("_", ""));
            details[1] = int(age_gender_color_by_regex.group(2).replace("_", ""));
            num_of_color = int(age_gender_color_by_regex.group(3).replace("_", ""));
            details[num_of_color + 2] = int(1);
            batch_details.append(details)
        batch[0] = batch_details

def split_train_test(full_dataset, train_percent = 0.7, validation_percent = 0.15, test_percent = 0.15):

    '''
    The function splits the list into three lists (train, validation and test) by the given percents.

    :Arg
        :param full_dataset (list): List that want to split
        :param train_percent (double): Number in [0,1]. The percent of information assigned to training
        :param validation_percent (double): Number in [0,1]. The percent of information assigned to validation
        :param test_percent (double): Number in [0,1]. The percent of information assigned to test
    :Returns
        list: three lists representing the train, validation and test
    '''

    # Number of features
    dataset_size = len(full_dataset)

    train_size = math.ceil(train_percent * dataset_size)
    test_validation_size = dataset_size - train_size
    validation_size = math.ceil(validation_percent/(1-train_percent) * test_validation_size)
    test_size = test_validation_size - validation_size

    train_index = train_size
    validation_index = train_index + validation_size
    test_index = dataset_size

    cutting_points = [train_index,validation_index]

    train_data = full_dataset[: train_index]
    validation_data = full_dataset[train_index : validation_index]
    test_data = full_dataset[validation_index :]

    return train_data, validation_data, test_data

def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

def encoder(X_in, X_details, keep_prob):

    '''
        The function builds the encoder

    :Args
        :param X_in: batch of pixesl e.g. [ [R1,G1,B1],[R2,G2,B2],[R3,G3,B3] ]
        :param X_details: details of images
        :param keep_prob: dropout probability

    :Returns
        tensor of mean, std, code of bottleneck
    '''
    activation = lrelu
    with tf.variable_scope("encoder", reuse=None):
        X = tf.reshape(X_in, shape=[-1, heigh_global, width_global, 3], name="encoder_input_reshape")
        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation, name="encoder_x_1")
        x = tf.nn.dropout(x, keep_prob)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="encoder_x_max_pool")
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation, name="encoder_x_2")
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation, name="encoder_x_3")
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        mn = tf.layers.dense(x, units=n_latent, name="encoder_mean")
        sd = 0.5 * tf.layers.dense(x, units=n_latent, name="encoder_std")
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]), name="encoder_epsilon")
        z = mn + tf.multiply(epsilon, tf.exp(sd), name="encoder_z")
        z = tf.concat([z, X_details], 1, name="encoder_z_concut")

        return z, mn, sd

def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu, name="decoder_x_1")
        x = tf.layers.dense(x, units=inputs_decoder * 2, activation=lrelu, name="decoder_x_2")
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu, name="decoder_x_3")
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu, name="decoder_x_4")
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu, name="decoder_x_5")

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=heigh_global * width_global * 3, activation=tf.nn.sigmoid, name="decoder_x_6")
        img = tf.reshape(x, shape=[-1, heigh_global, width_global, 3], name="decoder_img")
        return img

if __name__ == '__main__':

    # Read and analyze dataset

    # image_folder = 'Images/Test_images_2/'
    image_folder = 'Images/Test_single_image/'
    image_type = '*.jpg'
    batch_size = 1


    if not os.path.isdir(image_folder):
        raise Exception('Invalid folder path')

    name_files = []

    #
    batches, no_enter_to_batch = get_batches(image_folder, image_type, batch_size, True)

    random.seed(2019)
    random.shuffle(batches)

    extract_details_from_file_name(batches)

    # Training

    tf.reset_default_graph()

    # batch_size = 64
    X_in = tf.placeholder(dtype=tf.float32, shape=[None, heigh_global, width_global, 3], name='X')
    X_detail = tf.placeholder(dtype=tf.float32, shape=[None, 7], name='Details')
    Y = tf.placeholder(dtype=tf.float32, shape=[None, heigh_global, width_global, 3], name='Y')
    Y_flat = tf.reshape(Y, shape=[-1, heigh_global * width_global * 3])
    keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')


    dec_in_channels = 3 # RGB - 3 Grayscale - 1
    n_latent = 48 # the size of the code of the bottleneck

    reshaped_dim = [-1, 7, 7, dec_in_channels]
    inputs_decoder = 49 * dec_in_channels / 2

    sampled, mean, std = encoder(X_in, X_detail, keep_prob)

    dec = decoder(sampled, keep_prob)

    unreshaped = tf.reshape(dec, [-1, heigh_global*width_global * 3], name="img_flatten") # flatten the code
    img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1, name="img_loss")
    latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * std - tf.square(mean) - tf.exp(2.0 * std), 1, name="latent_loss")
    loss = tf.reduce_mean(img_loss + latent_loss, name="loss")
    optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)

    saver = tf.train.Saver()

    gender = input("Choose the gender (0 - male, 1 - female): ")
    years_between_images = input("Choose years range between image and image (5 - 50): ")
    body_color = input("Choose a body color (0 - 4): ")

with tf.Session() as sess:

    for i in range(len(batches)):

        batch = batches[i][1]  # Get pixels matrix of current batch from batches list
        details = np.array(batches[i][0])
        details = details.reshape([-1, 7])

        batch_img = [batch[0] for _ in range(1, 101, int(years_between_images))]
        # details_img = [[year, gender, 1, 0, 0, 0, 0] for year in range(1, 101, int(years_between_images))]
        details_img = []
        for year in range(1, 101, int(years_between_images)):
            details_for_one_image = [0] * 7 # Create empty list with zeros (Format [age, gender, color1, color2, color3, color4, color5]

            details_for_one_image[0] = year
            details_for_one_image[1] = gender
            details_for_one_image[int(body_color) + 2] = 1
            details_img.append(details_for_one_image)

        # Restore the saved vairable
        saver.restore(sess, './model_537.49414/saved_variable_VAE_Faces_56')
        # saver.restore(sess, './saved_variable_VAE_Faces_56')
        # Print the loaded variable
        imgs = np.reshape(np.array(sess.run([unreshaped], feed_dict={X_in: batch_img, X_detail: details_img, keep_prob: 1.0})), [-1,56,56,3])

        for j in range(len(batch_img)):
            im = np.array(imgs[j])
            im = im * 255.0
            f = np.array(im)
            img = smp.toimage(f)
            img.show()
