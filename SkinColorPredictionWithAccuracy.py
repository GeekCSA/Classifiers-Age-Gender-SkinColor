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
            index = (int(age_gender_color_by_regex.group(3).replace("_", "")))
            details = [0.] * 5  # Create a one-hot that contains zero (only for skin color).
            details[index]=1.0  #
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
        logits = tf.layers.dense(x, units=5, name="encoder_dense_5")



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

    #
    batches, no_enter_to_batch = get_batches(image_folder, image_type, batch_size, True)

    random.seed(2019)
    random.shuffle(batches)

    extract_details_from_file_name(batches)

    x_train, x_validation, x_test = split_train_test(batches, 0.80, 0.13, 0.07)

    # Training

    tf.reset_default_graph()
    categories = 5 #for soft max

    # features= [heigh_global, width_global, 1]
    X_in = tf.placeholder(dtype=tf.float32, shape=[None, heigh_global, width_global, 1], name='X')
    X_skin = tf.placeholder(dtype=tf.float32, shape=[None, categories], name='true_gender')  # 0 - men , 1-wonem
    # weights = tf.Variable(tf.truncated_normal([1, categories]))
    # biases = tf.Variable(tf.zeros([categories]))

    keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

    dec_in_channels = 1  # RGB - 3 Grayscale - 1

    skin = encoder(X_in, keep_prob)

    # age_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=X_age, logits=logits))
    # age_loss = tf.reduce_mean(-tf.reduce_sum(X_age * tf.log(age), reduction_indices=[1]))
    #gender_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=X_gender, logits=gender))

    skin=tf.nn.softmax(skin)
    skin_loss = tf.reduce_mean(-tf.reduce_sum(X_skin * tf.log(skin), reduction_indices=[1]))

    #gender_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=X_gender, logits=gender)
    #gender_loss = tf.reduce_mean(gender_loss)

    optimizer = tf.train.AdamOptimizer(0.0005).minimize(skin_loss)

    correct_prediction = tf.equal(tf.argmax(X_skin,1),tf.argmax(skin,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:

        # initialize all of the variables in the session
        sess.run(tf.global_variables_initializer())

        k_train = 0
        k_test = 0
        local_loss = 700
        number_of_ephochs = 337

        max_acc = 0.6
        coffedens = 1.06
        for ephoch in range(number_of_ephochs):

            total_acc_per_model = 0.0

            #train the model

            for i in range(len(x_train)):
                batch_train = np.array(x_train[i][1])  # Get pixels matrix of current batch from batches list
                batch_train = batch_train.reshape([-1, 56, 56, 1])
                details_train = np.array(x_train[i][0])
                details_train = details_train.reshape([-1, categories])

                sess.run(optimizer, feed_dict={X_in: batch_train, X_skin: details_train, keep_prob: 0.8})

                loss, skins, local_acc = sess.run([skin_loss, skin, accuracy],
                                                    feed_dict={X_in: batch_train, X_skin: details_train,
                                                               keep_prob: 1.0})

                total_acc_per_model += local_acc

            avg_local_acc = total_acc_per_model / (len(x_train))
            print("accuracy of train for ephoch number ", ephoch, " is: ", avg_local_acc)

            # Check the model on validation data
            total_acc_per_model = 0
            for i in range(len(x_validation)):
                batch_validation = np.array(x_validation[i][1])  # Get pixels matrix of current batch from batches list
                batch_validation= batch_validation.reshape([-1, 56, 56, 1])
                details_validation = np.array(x_validation[i][0])
                details_validation = details_validation.reshape([-1, categories])

                loss, skins, local_acc = sess.run([skin_loss, skin, accuracy],
                                                    feed_dict={X_in: batch_validation, X_skin: details_validation,
                                                               keep_prob: 1.0})

                total_acc_per_model += local_acc

            avg_local_acc = total_acc_per_model / (len(x_validation))
            print("\taccuracy of validation for ephoch number ", ephoch, " is: ", avg_local_acc)
            # print("Predicted skin: ")
            # print(skins)
            # print("\n")
            # print("Real skin: ")
            # print(details_validation)
            # print("\n")


            # Save the model if it better than prev saved model

            if avg_local_acc > max_acc * coffedens:
                if avg_local_acc > 0.85:
                    coffedens = 1.009
                if avg_local_acc > 0.95:
                    coffedens = 1.007

                max_acc = avg_local_acc

                os.mkdir(str(max_acc) + "_Skin_prediction_ephoch_"+str(ephoch))
                # Save the variable in the file
                saved_path = saver.save(sess, str(max_acc) + "_Skin_prediction" + '/saved_variable')
                print('model saved in {}'.format(saved_path))


        # for i in range(80001):
        #     if k_train == len(x_train):
        #         k_train = 0
        #
        #     batch_train = np.array(x_train[k_train][1])  # Get pixels matrix of current batch from batches list
        #     batch_train = batch_train.reshape([-1, 56, 56, 1])
        #     details_train = np.array(x_train[k_train][0])
        #     details_train = details_train.reshape([-1, categories])
        #
        #     sess.run(optimizer, feed_dict={X_in: batch_train, X_gender: details_train, keep_prob: 0.8})
        #
        #     if i % 237 == 0: #237*batch=all data
        #
        #         if k_test == len(x_test):
        #             k_test = 0
        #
        #         batch_test = np.array(x_test[k_test][1])  # Get pixels matrix of current batch from batches list
        #         batch_test = batch_test.reshape([-1, 56, 56, 1])
        #         details_test = np.array(x_test[k_test][0])
        #         details_test = details_test.reshape([-1, categories])
        #
        #         acc = 0
        #         curr_batch = 0
        #         for i in range(batch_size):
        #                 acc += accuracy.eval(feed_dict={x: batch_xs, y: batch_ys})
        #
        #             loss, genders, acc = sess.run([gender_loss, gender, accuracy],
        #                                       feed_dict={X_in: batch_test, X_gender: details_test, keep_prob: 1.0})
        #
        #         print("step %d, training accuracy %g" % (ephoch, acc / curr_batch))
        #
        #         print("Predicted gender: ")
        #         print(genders)
        #         print("\n")
        #         print("Real gender: ")
        #         print(details_test)
        #         print("\n")
        #         print("accuracy: ")
        #         print(acc)
        #         print("\n")
        #
        #         print('Iteration:', i, ' loss_test:',
        #               gender_loss.eval(feed_dict={X_in: batch_test, X_gender: details_test, keep_prob: 1.0}))
        #
        #         print('Iteration:', i, ' loss_train:',
        #               gender_loss.eval(feed_dict={X_in: batch_train, X_gender: details_train, keep_prob: 1.0}))
        #         print("\n\n\n")
        #         k_test += 1
        #
        #         if local_loss > loss * 1.4 and loss < 0.01:
        #             os.mkdir(str(loss) + "_Genders_prediction")
        #             # Save the variable in the file
        #             saved_path = saver.save(sess, str(loss) + "_Genders_prediction" + '/saved_variable')
        #             print('model saved in {}'.format(saved_path))
        #
        #             local_loss = loss
        #
        #     k_train += 1

        # # Save the variable in the file
        # os.mkdir("_Genders_prediction")
        # # Save the variable in the file
        # saved_path = saver.save(sess, str(loss) + "_Genders_prediction" + '/saved_variable')
        # print('model saved in {}'.format(saved_path))
        #
        # saved_path = saver.save(sess, './saved_variable')
        # print('model saved in {}'.format(saved_path))