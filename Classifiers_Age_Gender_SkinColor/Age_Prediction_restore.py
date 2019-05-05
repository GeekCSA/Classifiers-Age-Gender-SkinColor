import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import re
import os
import glob
import math
import random
import cv2

heigh_global = 56
width_global = 56




def get_batches(folder_path, image_type, size_batch=1, last_throw=False):
    '''
  The function groups the images into a batch, each containing 'batch_size' images
  :Args
      :param folder_path (str): The folder path of the images
      :param image_type (str): The type of the images (format: '*.png', '*.jpg' etc.)
      :param size_batch (int): The size of each batch
      :param last_throw (bool): If throw the last batch if it is not full
  :Returns
      list: List of batches, each batch holds #size_batch names (of files) and #size_batch images (pixels)
      int: The number of images in the last batch (if it is not full, if full then the number is 0)
  :Raises
      ValueError: If the value of size_batch is less than 1
  '''

    if size_batch < 1:
        raise ValueError('The size_batch should be greater than 0. The value of size_batch was: {}'.format(size_batch))

    num_of_images = len(next(os.walk(folder_path))[2])

    files_name = os.path.join(folder_path, image_type)

    images_name = [file_path for file_path in glob.iglob(files_name)]  # List of files name

    images_pixels = [cv2.imread(file_path, 0) for file_path in images_name]  # List of pixels
    images_pixels = np.array(images_pixels)
    images_pixels = images_pixels / 255.0

    batches = []

    for i in range(0, num_of_images - size_batch, size_batch):
        batch = [images_name[i: i + size_batch], images_pixels[i: i + size_batch]]
        batches.append(batch)

    no_enter_to_batch = num_of_images - len(batches) * size_batch

    if ((not last_throw and no_enter_to_batch != 0) or no_enter_to_batch == size_batch):
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
  The details contain age, gender and skin color, but we want only age becouse this model prediction age
  :Args
      :param batches (list): The output from "get_batches" function
  '''

    for batch in batches:
        batch_details = []
        for name in batch[0]:
            age_gender_color_by_regex = re.search('([0-9]{1,3})_([0-1])_([0-4])', name)
            #    details = [0.] * categories  # Create a one-hot that contains zero (only for age).
            #    index = mapAgeIntoClass(int(age_gender_color_by_regex.group(1).replace("_", ""))) #softmaax
            index = (int(age_gender_color_by_regex.group(1).replace("_", "")))  # linear
            details = index
            batch_details.append(details)
        batch[0] = batch_details


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def split_train_test(full_dataset, train_percent=0.7, validation_percent=0.15, test_percent=0.15):
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
    validation_size = math.ceil(validation_percent / (1 - train_percent) * test_validation_size)
    test_size = test_validation_size - validation_size

    train_index = train_size
    validation_index = train_index + validation_size
    test_index = dataset_size

    cutting_points = [train_index, validation_index]

    train_data = full_dataset[: train_index]
    validation_data = full_dataset[train_index: validation_index]
    test_data = full_dataset[validation_index:]

    return train_data, validation_data, test_data


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

def logistic_loss(pred, rael):
    eps = 1e-12
    return (-rael * np.log(pred+eps) - (1 - rael) * np.log(1 - pred+eps)).mean()

def encoder(X_in, keep_prob):
    '''
      The function builds the encoder
  :Args
      :param X_in: batch of pixesl e.g. [ [R1,G1,B1],[R2,G2,B2],[R3,G3,B3] ]
      :param keep_prob: dropout probability
  :Returns
      tensor of mean, std, code of bottleneck
  '''
    activation = lrelu

    with tf.variable_scope("encoder", reuse=None):
        X = tf.reshape(X_in, shape=[-1, heigh_global, width_global, 1], name="encoder_input_reshape")
        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation,
                             name="encoder_x_1")
        x = tf.nn.dropout(x, keep_prob)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="encoder_x_max_pool")
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation,
                             name="encoder_x_2")
        x = tf.nn.dropout(x, keep_prob)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="encoder_x_max_pool")
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation,
                             name="encoder_x_3")
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=4000, name="encoder_dense_4000")
        x = tf.layers.dense(x, units=1000, name="encoder_dense_1000")
        x = tf.layers.dense(x, units=250, name="encoder_dense_250")
        x = tf.layers.dense(x, units=50, name="encoder_dense_50")
        x = tf.layers.dense(x, units=25, name="encoder_dense_25")
        x = tf.layers.dense(x, units=12, name="encoder_dense_12")
        logits = tf.layers.dense(inputs=x, units=1)
        # age = tf.nn.softmax(logits)

        return logits

def run_model_on_data(sess, data):
    total_acc_per_model = 0.0


    return total_acc_per_model


if __name__ == '__main__':
    image_folder = 'Images/w56h56/'
    image_type = '*.jpg'
    batch_size = 100

    if not os.path.isdir(image_folder):
        raise Exception('Invalid folder path')

    name_files = []

    batches, no_enter_to_batch = get_batches(image_folder, image_type, batch_size, True)

    random.seed(2019)
    random.shuffle(batches)

    extract_details_from_file_name(batches)

    tf.reset_default_graph()
    categories = 1  # for linear

    # features= [heigh_global, width_global, 1]
    X_in = tf.placeholder(dtype=tf.float32, shape=[None, heigh_global, width_global, 1], name='X')
    X_age = tf.placeholder(dtype=tf.float32, shape=[None, categories], name='true_age')  # 1 - only age
    # weights = tf.Variable(tf.truncated_normal([1, categories]))
    # biases = tf.Variable(tf.zeros([categories]))

    keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

    dec_in_channels = 1  # RGB - 3 Grayscale - 1

    age = encoder(X_in, keep_prob)
    # age=tf.nn.softmax(tf.matmul(X_in,W)+b)

    # logits = tf.matmul(age, weights) + biases

    # age_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=X_age, logits=logits))
    # age_loss = tf.reduce_mean(-tf.reduce_sum(X_age * tf.log(age), reduction_indices=[1]))
    diff = np.subtract(age, X_age)
    age_loss = tf.reduce_mean(tf.pow(diff, 2), name="age_loss")

    optimizer = tf.train.AdamOptimizer(0.001).minimize(age_loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        saver.restore(sess, './45.5745_Age_prediction/saved_variable')
        print("Model restored.")

        # initialize all of the variables in the session
       # sess.run(tf.global_variables_initializer())


            # Check the model on test data
        for i in range(len(batches)):
            batch_test = np.array(batches[i][1])  # Get pixels matrix of current batch from batches list
            batch_test = batch_test.reshape([-1, 56, 56, 1])
            details_test = np.array(batches[i][0])
            details_test = details_test.reshape([-1, categories])

            loss, ages = sess.run([age_loss, age],
                                  feed_dict={X_in: batch_test, X_age: details_test, keep_prob: 1.0})

            print("Predicted ages: ")
            print(ages)
            print("\n")
            print("Real ages: ")
            print(details_test)
            print("\n")

        print("loss:")
        print(loss)
        print("\n")


