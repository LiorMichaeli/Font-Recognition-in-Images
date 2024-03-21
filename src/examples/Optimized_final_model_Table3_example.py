"""
This file contains the function that shows the results of the optimized final models from table 3 in the report

Authors:
    - Lior Michaeli
"""


# Import libraries
import torch

# Import files
from models.optimized_models import OptimizeResNetModel, Optimization_Handler
from config import config

def show_results_of_Optimized_final_model_Table3(train_and_val_loader, test_loader):
    """
    This function shows the results of the optimized final models from table 2 in the report:
    - The results on the training and validation set without smart algorithms,
      with first smart algorithm and with the second smart algorithm
    - Create the submit file for the test set in ../models/models_predictions_on_test/Optimized_Custom_ResNet_Table3_1, 
        ../models/models_predictions_on_test/Optimized_Custom_ResNet_Table3_2 and
        ../models/models_predictions_on_test/Optimized_Custom_ResNet_Table3_3
    """
    # Table 3 - 1
    Optimized_final_model_Table3_1 = OptimizeResNetModel(None, params={
        "number_of_ResNet_blocks": 3,
        "starting_number_of_channels": 16,
        "number_of_fc_layers": 1,
        "fc1_out_features": 3297
    })
    Optimized_final_model_Table3_1 = Optimized_final_model_Table3_1.to(config.MEMORY_DEVICE)
    Optimized_final_model_Table3_1.load_state_dict(torch.load("../models/models_weights/Optimized_final_model_Table3_1-production_model.pth", map_location=config.MEMORY_DEVICE))
    
    Optimized_final_model_Table3_1.create_submit_file(test_loader, "../models/models_weights/Optimized_final_model_Table3_1-production_model.pth",
                                                        "../models/models_predictions_on_test/Optimized_final_model_Table3_1-production_model")
    
    Optimized_final_model_Table3_1.show_all_the_results(train_and_val_loader, val_dataloader=None)

    # Table 3 - 2
    Optimized_final_model_Table3_2 = OptimizeResNetModel(None, params={
        "number_of_ResNet_blocks": 3,
        "starting_number_of_channels": 16,
        "number_of_fc_layers": 1,
        "fc1_out_features": 3297
    })
    Optimized_final_model_Table3_2 = Optimized_final_model_Table3_2.to(config.MEMORY_DEVICE)
    Optimized_final_model_Table3_2.load_state_dict(torch.load("../models/models_weights/Optimized_final_model_Table3_2.pth", map_location=config.MEMORY_DEVICE))
    
    Optimized_final_model_Table3_2.create_submit_file(test_loader, "../models/models_weights/Optimized_final_model_Table3_2.pth",
                                                        "../models/models_predictions_on_test/Optimized_final_model_Table3_2")
    
    Optimized_final_model_Table3_2.show_all_the_results(train_and_val_loader, val_dataloader=None)

    # Table 3 - 3
    Optimized_final_model_Table3_3 = OptimizeResNetModel(None, params={
        "number_of_ResNet_blocks": 3,
        "starting_number_of_channels": 16,
        "number_of_fc_layers": 1,
        "fc1_out_features": 3297
    })
    Optimized_final_model_Table3_3 = Optimized_final_model_Table3_3.to(config.MEMORY_DEVICE)
    Optimized_final_model_Table3_3.load_state_dict(torch.load("../models/models_weights/Optimized_final_model_Table3_3.pth", map_location=config.MEMORY_DEVICE))
    
    Optimized_final_model_Table3_3.create_submit_file(test_loader, "../models/models_weights/Optimized_final_model_Table3_3.pth",
                                                        "../models/models_predictions_on_test/Optimized_final_model_Table3_3")
    
    Optimized_final_model_Table3_3.show_all_the_results(train_and_val_loader, val_dataloader=None)


def optimize_final_model_example(train_loader, train_loader_for_evaluation):
    """
    This function optimizes the final model and saves the best model in ../models/models_weights/Optimized_final_model
    """
    optimizer_resnet_model = Optimization_Handler(train_loader, None, train_loader_for_evaluation, "../models/models_weights/Optimized_final_model")
    optimizer_resnet_model.optimize_model(number_of_trials=100, final_optimization=True)
    optimizer_resnet_model.show_results_of_spesific_trial(0, "../models/models_weights/Optimized_final_model_best_model_1.pth", is_final_optimization=True)