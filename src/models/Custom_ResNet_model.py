"""
A module for the Custom ResNet model

This module contains the Custom ResNet model

Authors:
    - Lior Michaeli
"""


# Import libraries
import torch
import torch.nn as nn


# Import files
from config import config
from .Base_model import BaseModel

class CustomResNetModel(BaseModel):
    """
    ResNet model

    :Attributes:

    History lists of model performance:
    - **train_loss_history** (list) - List of training loss history
    - **val_loss_history** (list) - List of validation loss history
    - **train_accuracy_without_smart_algorithms_history** (list) - List of training accuracy without smart algorithms history
    - **train_accuracy_with_first_smart_algorithm_history** (list) - List of training accuracy with first smart algorithm history
    - **train_accuracy_with_second_smart_algorithm_history** (list) - List of training accuracy with second smart algorithm history
    - **val_accuracy_without_smart_algorithms_history** (list) - List of validation accuracy without smart algorithms history
    - **val_accuracy_with_first_smart_algorithm_history** (list) - List of validation accuracy with first smart algorithm history
    - **val_accuracy_with_second_smart_algorithm_history** (list) - List of validation accuracy with second smart algorithm history

    Model layers:
    - **conv_start** (nn.Conv2d) - Convolutional layer
    - **conv1_block1** (nn.Conv2d) - Convolutional layer
    - **conv2_block1** (nn.Conv2d) - Convolutional layer
    - **conv_skip1** (nn.Conv2d) - Convolutional layer
    - **conv1_block2** (nn.Conv2d) - Convolutional layer
    - **conv2_block2** (nn.Conv2d) - Convolutional layer
    - **conv_skip2** (nn.Conv2d) - Convolutional layer
    - **bn2d_start** (nn.BatchNorm2d) - Batch Normalization layer
    - **bn2d_1_block1** (nn.BatchNorm2d) - Batch Normalization layer
    - **bn2d_2_block1** (nn.BatchNorm2d) - Batch Normalization layer
    - **bn2d_skip1** (nn.BatchNorm2d) - Batch Normalization layer
    - **bn2d_1_block2** (nn.BatchNorm2d) - Batch Normalization layer
    - **bn2d_2_block2** (nn.BatchNorm2d) - Batch Normalization layer
    - **bn2d_skip2** (nn.BatchNorm2d) - Batch Normalization layer
    - **fc1** (nn.Linear) - Fully connected layer
    - **fc2** (nn.Linear) - Fully connected layer
    - **fc3** (nn.Linear) - Fully connected layer
    - **relu** (nn.ReLU) - ReLU layer
    - **max_pool** (nn.MaxPool2d) - Max Pooling layer
    - **dropout1** (nn.Dropout) - Dropout layer
    - **dropout2** (nn.Dropout) - Dropout layer

    Loss function:
    - **loss_function** (nn.CrossEntropyLoss) - Loss function

    :Methods:

    - **__init__()** - Constructor
    - **forward(x)** - Forward pass of the model
    - **train_model(train_dataloader, val_dataloader, train_dataloader_for_evaluation, best_model_filename, num_epochs, learning_rate, lambd, do_learning_rate_decay, learning_rate_decay_factor, learning_rate_decay_step)** - Train the model
    - **compute_model_loss_on_dataset(dataloader)** - Compute the model loss on the dataset
    - **get_labels_of_dataset(dataloader)** - Get the labels of the dataset
    - **get_predictions_and_labels_of_dataset(dataloader, apply_first_smart_algorithm, apply_second_smart_algorithm)** - Get the predictions of the model on the dataset and labels of the dataset
    - **get_predictions_according_predictions_probs(dataloader, predictions_probs, apply_first_smart_algorithm, apply_second_smart_algorithm)** - Get the predictions of the model on the dataset according to the predictions probabilities
    - **plot_loss_and_accuracy_history()** - Plot the loss and accuracy history of the model
    - **plot_loss_history(train_loss_history, val_loss_history, plotting_title, axis)** - Plot the loss history of the model on the train and val dataset
    - **plot_accuracy_history(train_accuracy_history, val_accuracy_history, plotting_title, axis)** - Plot the accuracy history of the model on the train and val dataset
    - **plot_evaluation_graphs(dataloader, apply_first_smart_algorithm, apply_second_smart_algorithm)** - Plot evaluation graphs of the model on the dataset
    - **plot_ROC_curve(labels_np, predictions_probs_np, axis)** - Plot the ROC curve of the model on the dataset
    - **plot_Precision_Recall_curve(labels_np, predictions_probs_np, axis)** - Plot the Precision Recall curve of the model on the dataset
    - **plot_confusion_matrix(labels_np, predictions_np, axis)** - Plot the confusion matrix of the model on the dataset
    - **predict_probs(inputs)** - Predict probabilities of the model on the input data
    - **get_predictions_of_dataset(dataloader, apply_first_smart_algorithm, apply_second_smart_algorithm)** - Get the predictions of the model on the dataset
    - **get_predictions_probs_of_dataset(dataloader)** - Get the predictions probabilities of the model on the dataset
    - **get_predictions_according_two_models(probs1, probs2, words_lens)** - Get the predictions of the model on the dataset according to the predictions probabilities
    - **get_accuracy_of_dataset_according_two_models(dataloader, probs1, probs2)** - Get the accuracy of the model on the dataset according to the predictions probabilities
    - **evaluate_model(train_dataloader, val_dataloader, apply_first_smart_algorithm, apply_second_smart_algorithm)** - Evaluate the model on the train and val dataset
    - **evaluate_model_on_dataset(dataloader, apply_first_smart_algorithm, apply_second_smart_algorithm)** - Evaluate the model on the dataset
    - **evaluate_model_on_specific_char_in_dataset(dataloader, char, apply_first_smart_algorithm, apply_second_smart_algorithm)** - Evaluate the model on the specific char in the dataset
    - **evaluate_model_on_each_char_in_dataset(dataloader, apply_first_smart_algorithm, apply_second_smart_algorithm)** - Evaluate the model on each char in the dataset
    - **create_submit_file(test_loader, model_weights_path, submit_file_name)** - Create a submit file of the model on the test dataset for the competition page in Kaggle
    - **calc_amount_of_parameters()** - Calculate the amount of parameters of the model
    - **calculate_accuracy(labels, predictions)** - Calculate the accuracy of the model on the dataset
    """

    def __init__(self):
        """
        Constructor

        :return: None
        """
        super(CustomResNetModel, self).__init__()

        # Define layers
        # Convolutional layers
        self.conv_start = nn.Conv2d(in_channels=config.INPUT_SHAPE[0], out_channels=16, kernel_size=3, stride=1, padding=1)

        self.conv1_block1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2_block1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv_skip1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)

        self.conv1_block2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_block2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_skip2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Batch Normalization layers
        self.bn2d_start = nn.BatchNorm2d(16)
        self.bn2d_1_block1 = nn.BatchNorm2d(16)
        self.bn2d_2_block1 = nn.BatchNorm2d(32)
        self.bn2d_skip1 = nn.BatchNorm2d(32)
        
        self.bn2d_1_block2 = nn.BatchNorm2d(32)
        self.bn2d_2_block2 = nn.BatchNorm2d(64)
        self.bn2d_skip2 = nn.BatchNorm2d(64)


        # Fully connected layers
        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, config.NUM_OF_CLASSES_IN_FONT_DATASET)

        # Layers that have not learnable parameters
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout layers
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        # Loss function
        self.loss_function = nn.CrossEntropyLoss()

        # History lists of model performance
        # Loss history
        self.train_loss_history = []
        self.val_loss_history = []
        
        # Accuracy history
        self.train_accuracy_without_smart_algorithms_history = []
        self.train_accuracy_with_first_smart_algorithm_history = []
        self.train_accuracy_with_second_smart_algorithm_history = []
        self.val_accuracy_without_smart_algorithms_history = []
        self.val_accuracy_with_first_smart_algorithm_history = []
        self.val_accuracy_with_second_smart_algorithm_history = []

    def forward(self, x):
        """
        Forward pass of the model

        :param x: input data
        :type x: torch.Tensor
        :return: output data
        :rtype: torch.Tensor
        """
        # Start Block
        x = self.conv_start(x)
        x = self.bn2d_start(x)
        x = self.relu(x)
        
        # First Block
        skip1 = self.conv_skip1(x)
        x = self.conv1_block1(x)
        x = self.bn2d_1_block1(x)
        x = self.relu(x)
        x = self.conv2_block1(x)
        x = torch.add(x, skip1)
        x = self.bn2d_2_block1(x)
        x = self.relu(x)

        # Second Block
        skip2 = self.conv_skip2(x)
        x = self.conv1_block2(x)
        x = self.bn2d_1_block2(x)
        x = self.relu(x)
        x = self.conv2_block2(x)
        x = torch.add(x, skip2)
        x = self.bn2d_2_block2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        
        x = x.view(x.size(0), -1)
        # First FC layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Second FC layer
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        # Third FC layer - Output layer
        x = self.fc3(x)
        return x