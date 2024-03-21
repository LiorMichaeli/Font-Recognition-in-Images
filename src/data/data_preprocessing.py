"""
This module is responsible for the data preprocessing.

Authors:
    - Lior Michaeli
"""

# Import libraries
import cv2 as cv
import numpy as np
import h5py


# Import constants
from config import config


def get_chars_from_img_by_bb(img, chars_bb, chars_imgs):    
    """
    Get the characters from the image by their bounding box

    :param img: The original image from which the characters will be extracted
    :param chars_bb: The characters bounding box
    :param chars_imgs: The characters images, which will be filled by the function

    :return: None
    """
    
    # Get the images by bounding box
    for i in range(chars_bb.shape[-1]):
        # Get the char image
        char_img = extract_char(img, chars_bb[:, :, i])
        char_img_np = np.array(char_img)
        
        # Determine the interpolation method based on whether we are enlarging or reducing the image
        if char_img_np.shape[0] < config.INPUT_SHAPE[1] or char_img_np.shape[1] < config.INPUT_SHAPE[2]:
            interpolation_method = cv.INTER_CUBIC
        else:
            interpolation_method = cv.INTER_AREA
        # Resize the image to be in the same size
        char_img_np = cv.resize(char_img_np, (config.INPUT_SHAPE[2], config.INPUT_SHAPE[1]), interpolation=interpolation_method)

        chars_imgs.append(char_img_np)


def extract_char(img, bbox):
    """
    Extract the character from the image using the bounding box

    :param img: The original image from which the character will be extracted
    :param bbox: The bounding box of the character
    """
    # Ensure bbox is of shape (4, 2)
    if bbox.shape != (4, 2):
        bbox = np.transpose(bbox).reshape(-1, 2)

    # Compute the width and height of the bounding box
    width = int(max(np.linalg.norm(bbox[config.TOP_LEFT] - bbox[config.TOP_RIGHT]),
                    np.linalg.norm(bbox[config.BOTTOM_RIGHT] - bbox[config.BOTTOM_LEFT])))
    height = int(max(np.linalg.norm(bbox[config.TOP_LEFT] - bbox[config.BOTTOM_LEFT]),
                     np.linalg.norm(bbox[config.TOP_RIGHT] - bbox[config.BOTTOM_RIGHT])))

    # Define the destination points for the homography
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')
    bbox = np.array(bbox, dtype='float32')
    
    # Compute the homography matrix
    M = cv.getPerspectiveTransform(bbox, dst_pts)

    # Apply the homography to the original image
    char_img = cv.warpPerspective(img, M, (width, height))

    return char_img


def get_chars_slices_from_img(data, img_name, chars_imgs, chars_txt, chars_fonts, words_lengths, is_train_data, is_validation_data):
    """
    Get the characters slices from the image

    :param data: The data from which the characters will be extracted
    :param img_name: The image name
    :param chars_imgs: The characters images, which will be filled by the function
    :param chars_txt: The characters text, which will be filled by the function
    :param chars_fonts: The characters fonts, which will be filled by the function
    :param words_lengths: The words lengths, which will be filled by the function
    :param is_train_data: Whether the data is a training data
    :param is_validation_data: Whether the data is a validation data

    :return: None
    """
    # Get the image, font, text and bounding box
    img = data["data"][img_name][:]
    font = data["data"][img_name].attrs["font"] if is_train_data or is_validation_data else None
    txt = data["data"][img_name].attrs["txt"]
    charBB = data["data"][img_name].attrs["charBB"]
    txt = data["data"][img_name].attrs["txt"]
    
    # Get the chars fonts if we have fonts
    if is_train_data or is_validation_data:
        for char_idx in range(charBB.shape[-1]):
            chars_fonts.append(config.map_dict_font_to_id[font[char_idx]])
    
    # Get the chars text
    for word_idx in range(len(txt)):
        for char_idx in range(len(txt[word_idx])):
            chars_txt.append(chr(txt[word_idx][char_idx]))

    # Get the words lengths
    for word in txt:
        words_lengths.append(len(word))

    # Get the chars images
    get_chars_from_img_by_bb(img, charBB, chars_imgs)


def get_data(data_dir, is_train_data, is_validation_data, number_of_orig_imgs_in_training_dataset):
    """
    Get the data from the data directory according if it is a training or validation or test data.

    :param data_dir: The data directory
    :param is_train_data: Whether the data is a training data
    :param is_validation_data: Whether the data is a validation data
    :param number_of_orig_imgs_in_training_dataset: The number of original images in the training dataset

    :return: chars_imgs, imgs_fonts if it is a training or validation data, img_names, imgs_txt, words_lengths
    """
    # Load the data
    data = h5py.File(data_dir, "r")

    # Get the image names
    img_names = list(data["data"].keys())

    # Get the images
    chars_imgs = []
    imgs_fonts = []
    imgs_txt = []
    words_lengths = []

    # If it is a training or validation data, get the images names according to the number of original images in the training dataset
    if is_train_data:
        img_names = img_names[:number_of_orig_imgs_in_training_dataset]
    elif is_validation_data:
        img_names = img_names[number_of_orig_imgs_in_training_dataset:]

    for img_name in img_names:
        get_chars_slices_from_img(data, img_name, chars_imgs, imgs_txt, imgs_fonts, words_lengths,
                                    is_train_data, is_validation_data,)


    # Convert the image names, fonts and txt lists to numpy arrays
    imgs_fonts = np.array(imgs_fonts)
    imgs_txt = np.array(imgs_txt)
    img_names = np.array(img_names)

    if is_train_data or is_validation_data:
        return (chars_imgs, imgs_fonts, img_names, imgs_txt, words_lengths)
    else:
        return (chars_imgs, img_names, imgs_txt, words_lengths)
