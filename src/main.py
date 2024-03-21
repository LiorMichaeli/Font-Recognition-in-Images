"""
The main file of the project

Authors:
    - Lior Michaeli
"""


# Import libraries
import torch


# Import files
from data.data_preprocessing import get_data
from data.dataset_handler import get_data_sets, get_data_loaders, compute_mean_and_std_of_train_data
from visualization.visualize_data import plot_data_distribution, plot_chars_dataset
from models.BaseLine_model import BaseLineModel
from models.Custom_ResNet_model import CustomResNetModel
from models.optimized_models import OptimizeResNetModel
from config import config 
from examples.BaseLine_model_example import show_results_of_BaseLine_model, training_BaseLine_model_example
from examples.Custom_ResNet_example import show_results_of_Custom_ResNet, training_Custom_ResNet_model_example
from examples.Optimized_Custom_ResNet_Table2_example import show_results_of_Optimized_Custom_ResNet_Table2, optimize_Custom_ResNet_model_example
from examples.Optimized_final_model_Table3_example import show_results_of_Optimized_final_model_Table3, optimize_final_model_example


def main():
    """
    The main function
    """
    
    # Get the data of the chars from the h5 file
    train_data_chars_imgs, train_data_chars_fonts, train_data_chars_img_names, train_data_chars_txt, train_word_lengths = get_data(
        "../data/train.h5", is_train_data=True, is_validation_data=False,
          number_of_orig_imgs_in_training_dataset=config.NUMBER_OF_ORIG_IMGS_IN_TRAINING_SET_OF_CHARS)
    total_train_and_val_data_chars_imgs, total_train_and_val_data_chars_fonts, total_train_and_val_data_chars_img_names, total_train_and_val_data_chars_txt, total_train_word_lengths = get_data(
        "../data/train.h5", is_train_data=True, is_validation_data=True,
          number_of_orig_imgs_in_training_dataset=config.TOTAL_NUMBER_OF_ORIG_IMGS)
    val_data_chars_imgs, val_data_chars_fonts, val_data_chars_img_names, val_data_chars_txt, val_word_lengths = get_data(
        "../data/train.h5", is_train_data=False, is_validation_data=True,
          number_of_orig_imgs_in_training_dataset=config.NUMBER_OF_ORIG_IMGS_IN_TRAINING_SET_OF_CHARS)
    test_data_chars_imgs, test_data_chars_img_names, test_data_chars_txt, test_word_lengths = get_data(
        "../data/test.h5", is_train_data=False, is_validation_data=False,
          number_of_orig_imgs_in_training_dataset=config.NUMBER_OF_ORIG_IMGS_IN_TRAINING_SET_OF_CHARS)

    """
    # Show the training, validation and test data
    
    plot_chars_dataset(train_data_chars_imgs, train_data_chars_fonts, train_data_chars_txt, train_word_lengths)
    plot_chars_dataset(val_data_chars_imgs, val_data_chars_fonts, val_data_chars_txt, val_word_lengths)
    plot_chars_dataset(test_data_chars_imgs, [], test_data_chars_txt, test_word_lengths)
    """

    """
    # Show the chars distribution in the training data, validation data and test data
    
    plot_data_distribution(train_data_chars_txt, 'Train dataset Character Distribution', 'Character', 'Frequency')
    plot_data_distribution(val_data_chars_txt, 'Validation dataset Character Distribution', 'Character', 'Frequency')
    plot_data_distribution(test_data_chars_txt, 'Test dataset Character Distribution', 'Character', 'Frequency')
    """

    """
    # Show the fonts distribution in the training data and validation data
    
    plot_data_distribution(train_data_chars_fonts, 'Train dataset Font Distribution', 'Font', 'Frequency',
                            [font_name + "(" + str(font_key) + ")" for font_key, font_name in config.map_dict_id_to_font.items()])
    plot_data_distribution(val_data_chars_fonts, 'Validation dataset Font Distribution', 'Font', 'Frequency',
                            [font_name + "(" + str(font_key) + ")" for font_key, font_name in config.map_dict_id_to_font.items()])
    """
    
    # Get the mean and std of the training data
    train_dataset_mean, train_dataset_std = compute_mean_and_std_of_train_data(train_data_chars_imgs, train_data_chars_fonts, train_data_chars_txt, train_word_lengths)

    # Get the data sets
    train_dataset, train_and_val_dataset, val_dataset, test_dataset, train_vis_dataset, val_vis_dataset = get_data_sets(train_data_chars_imgs, train_data_chars_fonts, train_data_chars_txt, train_word_lengths,
                                                                                               total_train_and_val_data_chars_imgs, total_train_and_val_data_chars_fonts, total_train_and_val_data_chars_txt, total_train_word_lengths,
                                                                                               val_data_chars_imgs, val_data_chars_fonts, val_data_chars_txt, val_word_lengths,
                                                                                               test_data_chars_imgs, test_data_chars_txt, test_word_lengths,
                                                                                               train_dataset_mean, train_dataset_std)
    
    # Get the data loaders
    train_loader, train_and_val_loader, train_and_val_loader_for_evaluation, train_loader_for_evaluation, val_loader, test_loader, train_vis_loader, val_vis_loader = get_data_loaders(train_dataset, train_and_val_dataset, val_dataset, test_dataset, train_vis_dataset, val_vis_dataset)

    # Load the final model for production and create submit file for him
    final_model = OptimizeResNetModel(None, params={
        "number_of_ResNet_blocks": 3,
        "starting_number_of_channels": 16,
        "number_of_fc_layers": 1,
        "fc1_out_features": 3297
    })
    final_model = final_model.to(config.MEMORY_DEVICE)
    final_model.load_state_dict(torch.load("../models/models_weights/Optimized_final_model_Table3_1-production_model.pth", map_location=config.MEMORY_DEVICE))
    final_model.create_submit_file(test_loader, "../models/models_weights/Optimized_final_model_Table3_1-production_model.pth",
                                    "../models/models_predictions_on_test/Optimized_final_model_Table3_1-production_model")
    """
    Show the results of the final model for production

    final_model.show_all_the_results(train_and_val_loader_for_evaluation, val_dataloader=None)
    """
    
    """
    # Show results examples of the models from the report
    
    show_results_of_BaseLine_model(train_loader_for_evaluation, val_loader, test_loader)
    show_results_of_Custom_ResNet(train_loader_for_evaluation, val_loader, test_loader)
    show_results_of_Optimized_Custom_ResNet_Table2(train_loader_for_evaluation, val_loader, test_loader)
    show_results_of_Optimized_final_model_Table3(train_and_val_loader_for_evaluation, test_loader)
    """

    """
    # Examples of training and optimizing the models from the report
    
    training_BaseLine_model_example(train_loader, val_loader, train_loader_for_evaluation)
    training_Custom_ResNet_model_example(train_loader, val_loader, train_loader_for_evaluation)
    optimize_Custom_ResNet_model_example(train_loader, val_loader, train_loader_for_evaluation)
    optimize_final_model_example(train_and_val_loader, train_and_val_loader_for_evaluation)
    """
    

if __name__ == '__main__':
    main()