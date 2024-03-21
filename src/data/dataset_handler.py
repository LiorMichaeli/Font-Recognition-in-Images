"""
The dataset handler module

This module contains classes and functions for handling the data sets

Author:
    Lior Michaeli
"""


# Import libraries
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


# Import files
from config import config

def compute_mean_and_std_of_dataset(dataloader):
    """
    Compute the mean and std of the dataset
    :param dataloader: The data loader of the dataset
    :return: The mean and std of the dataset
    """
    # Calculate mean and std for each channel(in total there are 3 channels)
    mean = np.zeros(config.INPUT_SHAPE[0])
    std = np.zeros(config.INPUT_SHAPE[0])
    total_imgs = len(dataloader.dataset)

    for imgs, _, _ in dataloader:
        # Iterate imgs in the current mini batch
        for img in imgs:
            mean += np.mean(img.numpy(), axis=(1, 2))
            std += np.std(img.numpy(), axis=(1, 2))

    mean /= total_imgs
    std /= total_imgs

    return mean, std


def compute_mean_and_std_of_train_data(train_data_chars_imgs_rgb, train_data_chars_fonts_rgb, train_data_chars_txt_rgb, train_word_lengths_rgb):
    """
    Compute the mean and std of the training data
    :param train_data_chars_imgs_rgb: The images of the training data
    :param train_data_chars_fonts_rgb: The fonts of the training data
    :param train_data_chars_txt_rgb: The text of the training data
    :param train_word_lengths_rgb: The lengths of the words of the training data
    :return: The mean and std of the training data
    """
    # Define transform to the data set for normalization computation
    normalization_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create the data set for normalization computation
    train_dataset_for_normalization_computation_rgb = CharsDataset(train_data_chars_imgs_rgb, train_data_chars_fonts_rgb, train_data_chars_txt_rgb, train_word_lengths_rgb, transforms=normalization_transforms)

    # Create the data loader for normalization computation
    train_loader_for_normalization_computation_rgb = DataLoader(train_dataset_for_normalization_computation_rgb, batch_size=config.BATCH_SIZE)

    # Compute the mean and std of the training dataset
    train_dataset_rgb_mean, train_dataset_rgb_std = compute_mean_and_std_of_dataset(train_loader_for_normalization_computation_rgb)

    return train_dataset_rgb_mean, train_dataset_rgb_std


class CharsDataset(torch.utils.data.Dataset):
    """
    A class for the characters data set

    :Attributes:
    - **imgs** (list of np.array): The images of the characters
    - **fonts** (list of np.array): The fonts of the characters
    - **txt** (list of str): The text of the characters
    - **transforms** (torchvision.transforms.Compose): The transforms to apply on the images
    - **words_lens** (list of int): The lengths of the words of the characters

    :Methods:
    - __len__: Returns the length of the data set
    - __getitem__: Returns the image, text of the character and font at the given index
    """
    def __init__(self, imgs, fonts, chars, words_lens, transforms=None):
        """
        :param imgs: The images of the characters
        :param fonts: The fonts of the characters
        :param chars: The text of the characters
        :param words_lens: The lengths of the words of the characters
        :param transforms: The transforms to apply on the images, default is None

        :return: None
        """
        self.imgs = imgs
        self.fonts = fonts
        self.chars = chars
        self.transforms = transforms
        self.words_lens = words_lens

    def __len__(self):
        """
        Returns the length of the data set
        """
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Returns the image, text of the character and font at the given index
        :param idx: The index of the data sample to return
        :return: The image, text of the character and font at the given index
        """
        # Check if the index is out of range
        if idx >= len(self.imgs):
            return None
        
        # Get the image, char and font and apply the transforms on the image, if we have transforms
        img, char, font = self.imgs[idx], self.chars[idx] if len(self.chars) > 0 else np.array([]), self.fonts[idx] if len(self.fonts) > 0 else np.array([])
        
        if self.transforms:
            img = self.transforms(img)
        return img, char, font

def get_data_sets(train_data_chars_imgs_rgb, train_data_chars_fonts_rgb, train_data_chars_txt_rgb, train_word_lengths_rgb,
                    total_train_and_val_data_chars_imgs, total_train_and_val_data_chars_fonts, total_train_and_val_data_chars_txt, total_train_word_lengths,
                      val_data_chars_imgs_rgb, val_data_chars_fonts_rgb, val_data_chars_txt_rgb, val_word_lengths_rgb,
                        test_data_chars_imgs_rgb, test_data_chars_txt_rgb, test_word_lengths_rgb,
                          train_dataset_rgb_mean, train_dataset_rgb_std):
    """
    Create the data sets
    :param train_data_chars_imgs_rgb: The images of the training data
    :param train_data_chars_fonts_rgb: The fonts of the training data
    :param train_data_chars_txt_rgb: The text of the training data
    :param train_word_lengths_rgb: The lengths of the words of the training data
    :param total_train_and_val_data_chars_imgs: The images of the training and validation data
    :param total_train_and_val_data_chars_fonts: The fonts of the training and validation data
    :param total_train_and_val_data_chars_txt: The text of the training and validation data
    :param total_train_word_lengths: The lengths of the words of the training and validation data
    :param val_data_chars_imgs_rgb: The images of the validation data
    :param val_data_chars_fonts_rgb: The fonts of the validation data
    :param val_data_chars_txt_rgb: The text of the validation data
    :param val_word_lengths_rgb: The lengths of the words of the validation data
    :param test_data_chars_imgs_rgb: The images of the test data
    :param test_data_chars_txt_rgb: The text of the test data
    :param test_word_lengths_rgb: The lengths of the words of the test data
    :param train_dataset_rgb_mean: The mean of the training data
    :param train_dataset_rgb_std: The std of the training data
    :return: The data sets
    """
    # Define transforms to the datasets
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(train_dataset_rgb_mean, train_dataset_rgb_std, inplace=True)
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(train_dataset_rgb_mean, train_dataset_rgb_std, inplace=True)
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(train_dataset_rgb_mean, train_dataset_rgb_std, inplace=True)
    ])
    vis_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create the datasets for the model
    train_dataset = CharsDataset(train_data_chars_imgs_rgb, train_data_chars_fonts_rgb, train_data_chars_txt_rgb, train_word_lengths_rgb, transforms=train_transforms)
    train_and_val_dataset = CharsDataset(total_train_and_val_data_chars_imgs, total_train_and_val_data_chars_fonts, total_train_and_val_data_chars_txt, total_train_word_lengths, transforms=train_transforms)
    val_dataset = CharsDataset(val_data_chars_imgs_rgb, val_data_chars_fonts_rgb, val_data_chars_txt_rgb, val_word_lengths_rgb, transforms=val_transforms)
    test_dataset = CharsDataset(test_data_chars_imgs_rgb, [], test_data_chars_txt_rgb, test_word_lengths_rgb, transforms=test_transforms)

    # Create the datasets for visualization
    train_vis_dataset = CharsDataset(train_data_chars_imgs_rgb, train_data_chars_fonts_rgb, train_data_chars_txt_rgb, train_word_lengths_rgb, transforms=vis_transforms)
    val_vis_dataset = CharsDataset(val_data_chars_imgs_rgb, val_data_chars_fonts_rgb, val_data_chars_txt_rgb, val_word_lengths_rgb, transforms=vis_transforms)

    return train_dataset, train_and_val_dataset, val_dataset, test_dataset, train_vis_dataset, val_vis_dataset
    

def get_data_loaders(train_dataset, train_and_val_dataset, val_dataset, test_dataset, train_vis_dataset, val_vis_dataset):
    """
    Create data loaders
    :param train_dataset: The training data set
    :param train_and_val_dataset: The training and validation data set
    :param val_dataset: The validation data set
    :param test_dataset: The test data set
    :param train_vis_dataset: The training data set for visualization
    :param val_vis_dataset: The validation data set for visualization
    :return: The data loaders
    """
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    train_and_val_loader = DataLoader(train_and_val_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    train_and_val_loader_for_evaluation = DataLoader(train_and_val_dataset, batch_size=config.BATCH_SIZE)
    train_loader_for_evaluation = DataLoader(train_dataset, batch_size=config.BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    # Create data loaders for visualization
    train_vis_loader = DataLoader(train_vis_dataset, batch_size=config.BATCH_SIZE)
    val_vis_loader = DataLoader(val_vis_dataset, batch_size=config.BATCH_SIZE)

    return train_loader, train_and_val_loader, train_and_val_loader_for_evaluation, train_loader_for_evaluation, val_loader, test_loader, train_vis_loader, val_vis_loader