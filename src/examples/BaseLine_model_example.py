"""
This module contains the function that shows the results of the BaseLine model from the report

Authors:
    - Lior Michaeli
"""


# Import libraries
import torch


# Import files
from models.BaseLine_model import BaseLineModel
from config import config


def show_results_of_BaseLine_model(train_loader_for_evaluation, val_loader, test_loader):
    """
    This function shows the results of the BaseLine model from the report:
    - The results on the training and validation sets without smart algorithms,
      with first smart algorithm and with the second smart algorithm
    - Create the submit file for the test set in ../models/models_predictions_on_test/BaseLine
    """
    BaseLine_model = BaseLineModel()
    BaseLine_model = BaseLine_model.to(config.MEMORY_DEVICE)
    BaseLine_model.load_state_dict(torch.load("../models/models_weights/BaseLine_model.pth", map_location=config.MEMORY_DEVICE))
    BaseLine_model.create_submit_file(test_loader, "../models/models_weights/BaseLine_model.pth",
                                       "../models/models_predictions_on_test/BaseLine", apply_first_smart_algorithm=False, apply_second_smart_algorithm=False)

    BaseLine_model.show_all_the_results(train_loader_for_evaluation, val_loader)

def training_BaseLine_model_example(train_loader, val_loader, train_loader_for_evaluation):
    """
    This function trains the BaseLine model and saves the best model in ../models/models_weights/BaseLineExample.pth
    """
    BaseLine_model = BaseLineModel()
    BaseLine_model = BaseLine_model.to(config.MEMORY_DEVICE)

    best_model_path = "../models/models_weights/BaseLineExample.pth"
    BaseLine_model.train_model(train_loader, val_loader, train_loader_for_evaluation, best_model_path, num_epochs=30, learning_rate=5e-4, lambd=0, do_learning_rate_decay=True, learning_rate_decay_factor=0.2, learning_rate_decay_step=20)