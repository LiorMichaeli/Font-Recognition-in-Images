"""
This module contains the function that shows the results of the optimized custom resnet models from table 2 in the report

Authors:
    - Lior Michaeli
"""

# Import libraries
import torch


# Import files
from models.optimized_models import OptimizeResNetModel, Optimization_Handler
from config import config


def show_results_of_Optimized_Custom_ResNet_Table2(train_loader_for_evaluation, val_loader, test_loader):
    """
    This function shows the results of the optimized custom resnet models from table 2 in the report:
    - The results on the training and validation sets without smart algorithms,
      with first smart algorithm and with the second smart algorithm
    - Create the submit file for the test set in ../models/models_predictions_on_test/Optimized_Custom_ResNet_Table2_1, 
        ../models/models_predictions_on_test/Optimized_Custom_ResNet_Table2_2 and
        ../models/models_predictions_on_test/Optimized_Custom_ResNet_Table2_3
    """
    # Table 2 - 1
    Optimized_Custom_ResNet_Table2_1_model = OptimizeResNetModel(None, params={
        "number_of_ResNet_blocks": 3,
        "starting_number_of_channels": 16,
        "number_of_fc_layers": 1,
        "fc1_out_features": 3297
    })
    Optimized_Custom_ResNet_Table2_1_model = Optimized_Custom_ResNet_Table2_1_model.to(config.MEMORY_DEVICE)
    Optimized_Custom_ResNet_Table2_1_model.load_state_dict(torch.load("../models/models_weights/Optimized_Custom_ResNet_Table2_1.pth", map_location=config.MEMORY_DEVICE))
    
    Optimized_Custom_ResNet_Table2_1_model.create_submit_file(test_loader, "../models/models_weights/Optimized_Custom_ResNet_Table2_1.pth",
                                                               "../models/models_predictions_on_test/Optimized_Custom_ResNet_Table2_1")
    
    Optimized_Custom_ResNet_Table2_1_model.show_all_the_results(train_loader_for_evaluation, val_loader)

    # Table 2 - 2
    Optimized_Custom_ResNet_Table2_2_model = OptimizeResNetModel(None, params={
        "number_of_ResNet_blocks": 3,
        "starting_number_of_channels": 32,
        "number_of_fc_layers": 2,
        "fc1_out_features": 6335,
        "fc2_out_features": 2478
    })
    Optimized_Custom_ResNet_Table2_2_model = Optimized_Custom_ResNet_Table2_2_model.to(config.MEMORY_DEVICE)
    Optimized_Custom_ResNet_Table2_2_model.load_state_dict(torch.load("../models/models_weights/Optimized_Custom_ResNet_Table2_2.pth", map_location=config.MEMORY_DEVICE))
    
    Optimized_Custom_ResNet_Table2_2_model.create_submit_file(test_loader, "../models/models_weights/Optimized_Custom_ResNet_Table2_2.pth",
                                                               "../models/models_predictions_on_test/Optimized_Custom_ResNet_Table2_2")
    
    Optimized_Custom_ResNet_Table2_2_model.show_all_the_results(train_loader_for_evaluation, val_loader)

    # Table 2 - 3
    Optimized_Custom_ResNet_Table2_3_model = OptimizeResNetModel(None, params={
        "number_of_ResNet_blocks": 3,
        "starting_number_of_channels": 32,
        "number_of_fc_layers": 1,
        "fc1_out_features": 7727
    })
    Optimized_Custom_ResNet_Table2_3_model = Optimized_Custom_ResNet_Table2_3_model.to(config.MEMORY_DEVICE)
    Optimized_Custom_ResNet_Table2_3_model.load_state_dict(torch.load("../models/models_weights/Optimized_Custom_ResNet_Table2_3.pth", map_location=config.MEMORY_DEVICE))
    
    Optimized_Custom_ResNet_Table2_3_model.create_submit_file(test_loader, "../models/models_weights/Optimized_Custom_ResNet_Table2_3.pth",
                                                                "../models/models_predictions_on_test/Optimized_Custom_ResNet_Table2_3")
    
    Optimized_Custom_ResNet_Table2_3_model.show_all_the_results(train_loader_for_evaluation, val_loader)


def optimize_Custom_ResNet_model_example(train_loader, val_loader, train_loader_for_evaluation):
    """
    This function optimizes the Custom ResNet model and saves the best model in ../models/models_weights/Optimized_Custom_ResNet path
    """
    optimizer_resnet_model = Optimization_Handler(train_loader, val_loader, train_loader_for_evaluation, "../models/models_weights/Optimized_Custom_ResNet")
    optimizer_resnet_model.optimize_model(number_of_trials=100, final_optimization=False)
    optimizer_resnet_model.show_results_of_spesific_trial(0, "../models/models_weights/Optimized_Custom_ResNet_best_model_1.pth", is_final_optimization=False)