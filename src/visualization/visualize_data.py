"""
Visualize data module

This module contains functions to visualize the data

Authors:
    - Lior Michaeli
"""


# Imports libraries
import matplotlib.pyplot as plt
from collections import Counter


# Import files
from config import config

def plot_chars_dataset(chars_imgs, chars_fonts, chars_txt, word_lens):
    """
    Plots the characters dataset
    :param chars_imgs: list of characters images
    :param chars_fonts: list of characters fonts
    :param chars_txt: list of characters texts
    :param word_lens: list of word lengths
    :return: None

    :Example:

    chars_imgs = [char_img1, char_img2, char_img3, char_img4, char_img5, char_img6]
    chars_fonts = [0, 1, 2, 3, 4, 5]
    chars_txt = ['a', 'b', 'c', 'd', 'e', 'f']
    word_lens = [3, 3]
    plot_chars_dataset(chars_imgs, chars_fonts, chars_txt, word_lens)
    """
    curr_idx = 0
    for word_len in word_lens:
        fig, axs = plt.subplots(1, word_len, figsize=(5 * word_len, 3))
        # Check if the font is available
        if len(chars_fonts):
            # Go through the characters in the current word
            for idx, (char_img, char_font, char_txt) in enumerate(zip(chars_imgs[curr_idx:curr_idx + word_len],
                                                             chars_fonts[curr_idx:curr_idx + word_len],
                                                               chars_txt[curr_idx:curr_idx + word_len])):
                axs[idx].imshow(char_img, cmap='gray')
                axs[idx].set_title("Char: " + char_txt + ", Label: " + config.map_dict_id_to_font[char_font])
                axs[idx].axis('off')
        else:
            # Go through the characters in the current word
            for idx, (char_img, char_txt)  in enumerate(zip(chars_imgs[curr_idx:curr_idx + word_len],
                                                           chars_txt[curr_idx:curr_idx + word_len])):
                axs[idx].imshow(char_img, cmap='gray')
                axs[idx].set_title("Char: " + char_txt)
                axs[idx].axis('off')
        curr_idx += word_len
        plt.tight_layout()  # Adjust layout to minimize overlap
        plt.show()


def plot_data_distribution(data, title, xlabel, ylabel, x_values_names = None):
    """
    Plots the distribution of the data
    :param data: list of data
    :param title: title of the plot
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param x_values_names: list of names for the x values, if None, the x values will be the ids
    :return: None

    :Example:

    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    title = "Data distribution"
    xlabel = "Data"
    ylabel = "Frequency"
    plot_data_distribution(data, title, xlabel, ylabel)
    """
    # Count the occurrences of each element
    counts = Counter(data)
    counts = dict(sorted(counts.items()))

    # Calculate the number of unique characters in the dataset
    num_unique_values = len(counts)

    # Plot the distribution
    plt.figure(figsize=(18, 6))
    plt.bar(counts.keys(), counts.values())
    plt.title(title + " ( Total elements = " + str(len(data)) + ", unique values = " + str(num_unique_values) +
               ", min freq = " + str(min(counts.values())) + ", max freq = " + str(max(counts.values())) + ")")
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)

    # Replace the ids with their corresponding names
    if x_values_names is not None:
        plt.xticks(list(counts.keys()), x_values_names)


    plt.show()
    