import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    # Making lists to store the images and labels
    images = []
    labels = []
    # Getting the list of category folder in directory path of the data_dir
    pathDir = os.listdir(os.path.join(os.path.curdir, data_dir))
    for folder in pathDir:
        # Selecting only the folders which are our categories
        if folder.isnumeric():
            # Getting the list of images in the folder directory path
            pathImgDir = os.listdir(os.path.join(os.path.curdir, data_dir, folder))
            for img in pathImgDir:
                # Reading the image with OpenCV-python
                final = cv2.imread(os.path.join(os.path.curdir, data_dir, folder, img))
                # Resizing the image to IMG_WIDTH, IMG_HEIGHT, 3 (3 for RGB)
                final_img = cv2.resize(final, (IMG_WIDTH, IMG_HEIGHT), 3)
                # Adding the images & labels to their lists
                images.append(final_img)
                labels.append(int(folder))

    return images, labels

    raise NotImplementedError


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Create a neural network
    model = tf.keras.models.Sequential()

    # Adding convolution 2D layer with 32 features
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    # Adding a max-pooling layer of 2x2 matrix
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Adding 2nd convolution 2D layer with 32 features
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
    # Adding 2nd max-pooling layer of 2x2 matrix
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Adding a Flattening layer
    model.add(tf.keras.layers.Flatten())

    # Add a hidden layer with 256 units, with ReLU activation
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    # Add a dropout layer to avoid overfitting
    model.add(tf.keras.layers.Dropout(0.5))

    # Add output layer with NUM_CATEGORIES units, with softmax activation
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))

    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

    raise NotImplementedError

if __name__ == "__main__":
    main()
