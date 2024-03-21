"""
This file shows the results of the Custom ResNet model from the report

Authors:
    - Lior Michaeli
"""


# Import libraries
import torch


# Import files
from models.Custom_ResNet_model import CustomResNetModel
from config import config


def show_results_of_Custom_ResNet(train_loader_for_evaluation, val_loader, test_loader):
    """
    This function shows the results of the Custom ResNet model from the report:
    - The results on the training and validation data sets without smart algorithms,
      with first smart algorithm and with the second smart algorithm
    - Create the submit file for the test set in ../models/models_predictions_on_test/Custom_ResNet
    """
    Custom_ResNet_model = CustomResNetModel()
    Custom_ResNet_model = Custom_ResNet_model.to(config.MEMORY_DEVICE)
    Custom_ResNet_model.load_state_dict(torch.load("../models/models_weights/Custom_ResNet.pth", map_location=config.MEMORY_DEVICE))
    
    Custom_ResNet_model.create_submit_file(test_loader, "../models/models_weights/Custom_ResNet.pth",
                                            "../models/models_predictions_on_test/Custom_ResNet_without_smart_predictions",
                                              apply_first_smart_algorithm=False, apply_second_smart_algorithm=False)
    Custom_ResNet_model.create_submit_file(test_loader, "../models/models_weights/Custom_ResNet.pth",
                                            "../models/models_predictions_on_test/Custom_ResNet_with_first_smart_predictions",
                                              apply_first_smart_algorithm=True, apply_second_smart_algorithm=False)
    Custom_ResNet_model.create_submit_file(test_loader, "../models/models_weights/Custom_ResNet.pth",
                                            "../models/models_predictions_on_test/Custom_ResNet_with_second_smart_predictions",
                                              apply_first_smart_algorithm=False, apply_second_smart_algorithm=True)
    
    Custom_ResNet_model.show_all_the_results(train_loader_for_evaluation, val_loader)

def training_Custom_ResNet_model_example(train_loader, val_loader, train_loader_for_evaluation):
    """
    This function trains the Custom ResNet model and saves the best model in ../models/models_weights/Custom_ResNetExample.pth
    """
    Custom_ResNet_model = CustomResNetModel()
    Custom_ResNet_model = Custom_ResNet_model.to(config.MEMORY_DEVICE)

    best_model_path = "../models/models_weights/Custom_ResNetExample.pth"
    Custom_ResNet_model.train_model(train_loader, val_loader, train_loader_for_evaluation, best_model_path, num_epochs=30, learning_rate=1e-3, lambd=0, do_learning_rate_decay=True, learning_rate_decay_factor=0.1, learning_rate_decay_step=15)