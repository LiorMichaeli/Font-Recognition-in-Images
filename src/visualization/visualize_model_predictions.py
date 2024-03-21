"""
Visualize model predictions module

This module contains functions to visualize the model predictions

Authors:
    - Lior Michaeli
"""


# Import libraries
import matplotlib.pyplot as plt

# Import files
from ..config import config


def plot_model_predictions_on_dataset(data_loader, words_lens, labels, predictions_probs, predictions_without_smart_algos=None,
                                       predictions_with_first_smart_algo=None, predictions_with_second_smart_algo=None):
    """
    Plots the model predictions on the dataset
    :param data_loader: data loader
    :param words_lens: list of word lengths
    :param labels: list of labels
    :param predictions_probs: list of predictions probabilities
    :param predictions_without_smart_algos: list of predictions without smart algorithms, default is None
    :param predictions_with_first_smart_algo: list of predictions with first smart algorithm, default is None
    :param predictions_with_second_smart_algo: list of predictions with second smart algorithm, default is None
    :return: None

    :Example:

    words_lens = [3, 3]
    labels = [0, 1, 2, 3, 4, 5]
    predictions_probs = [word1_probs, word2_probs]
    predictions_without_smart_algos = [0, 1, 2, 3, 4, 5]
    predictions_with_first_smart_algo = [0, 1, 2, 3, 4, 5]
    predictions_with_second_smart_algo = [0, 1, 2, 3, 4, 5]
    plot_model_predictions_on_dataset(data_loader, words_lens, labels, predictions_probs, predictions_without_smart_algos, predictions_with_first_smart_algo, predictions_with_second_smart_algo)
    """
    curr_idx = 0

    for i in range(len(words_lens)):

        # Get word labels and predictions
        word_labels = labels[curr_idx:curr_idx + words_lens[i]]
        word_predictions_without_smart_algos = predictions_without_smart_algos[curr_idx:curr_idx + words_lens[i]] if predictions_without_smart_algos else None
        word_predictions_with_first_algo = predictions_with_first_smart_algo[curr_idx:curr_idx + words_lens[i]] if predictions_with_first_smart_algo else None
        word_predictions_with_second_algo = predictions_with_second_smart_algo[curr_idx:curr_idx + words_lens[i]] if predictions_with_second_smart_algo else None
        word_predictions_probs = predictions_probs[curr_idx:curr_idx + words_lens[i]]

        # Create a 2x2 grid of subplots
        fig, axs = plt.subplots(1, words_lens[i], figsize=(3 * words_lens[i], 2 * words_lens[i]))

        for j in range(words_lens[i]):
            axs[j].imshow(data_loader.dataset.imgs[curr_idx + j])
            axs[j].set_title("Char: " + str(data_loader.dataset.chars[curr_idx + j]) + "\n Label: " + config.map_dict_id_to_font[word_labels[j]]
                            + "\n Without Smart algos: " + config.map_dict_id_to_font[word_predictions_without_smart_algos[j]] if word_predictions_without_smart_algos else ""
                            + "\n First Smart algo: " + config.map_dict_id_to_font[word_predictions_with_first_algo[j]] if word_predictions_with_first_algo else ""
                            + "\n Second Smart algo: " + config.map_dict_id_to_font[word_predictions_with_second_algo[j]] if word_predictions_with_second_algo else ""
                            + "\n Prob: " + str(word_predictions_probs[j][word_predictions_probs[j]])) if word_predictions_probs else ""
            axs[j].axis('off')

        plt.show()