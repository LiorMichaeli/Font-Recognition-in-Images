"""
A module that contains the BaseModel class

This module contains the BaseModel class, which is an abstract class that represents a model

Authors:
    - Lior Michaeli
"""


# Imports libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, roc_curve
import seaborn as sns


# Import files
from config import config


class BaseModel(nn.Module):
    """
    A class used to represent a abstract model

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
        Constructs all the necessary attributes for the BaseModel object
        """
        super(BaseModel, self).__init__()

    def forward(self, x):
        """
        Forward pass of the model

        :param x: input data
        :type x: torch.Tensor
        :return: model predictions probabilities on the input data
        """
        raise NotImplementedError
    
    def train_model(self, train_dataloader, val_dataloader, train_dataloader_for_evaluation, best_model_filename,
                     num_epochs=10, learning_rate=0.01, lambd=0.0005,
                       do_learning_rate_decay=True, learning_rate_decay_factor=0.1, learning_rate_decay_step=15):
        # Define a optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=lambd)
        
        # Define a learning rate scheduler
        if do_learning_rate_decay:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=learning_rate_decay_step, gamma=learning_rate_decay_factor)
        
        best_model_val_accuracy = -float('infinity')

        for epoch in range(num_epochs):
            # Training
            train_total_epoch_loss = 0.0
            self.train()
            for train_inputs, _, train_labels in train_dataloader:
                train_inputs = train_inputs.to(config.MEMORY_DEVICE)
                train_labels = train_labels.to(config.MEMORY_DEVICE).long()
                optimizer.zero_grad()
                y_pred_logits = self(train_inputs)
                train_mini_batch_loss = self.loss_function(y_pred_logits, train_labels)
                train_total_epoch_loss += train_mini_batch_loss
                train_mini_batch_loss.backward()
                optimizer.step()
            train_loss = train_total_epoch_loss / len(train_dataloader)
            self.train_loss_history.append(train_loss)

            # Validation
            val_loss = self.compute_model_loss_on_dataset(val_dataloader)
            self.val_loss_history.append(val_loss)

            train_accuracy_not_smart, val_accuracy_not_smart = self.evaluate_model(train_dataloader, val_dataloader, apply_first_smart_algorithm=False, apply_second_smart_algorithm=False)
            train_accuracy_with_first_smart_algorithm, val_accuracy_with_first_smart_algorithm = self.evaluate_model(train_dataloader_for_evaluation, val_dataloader, apply_first_smart_algorithm=True, apply_second_smart_algorithm=False)
            train_accuracy_with_second_smart_algorithm, val_accuracy_with_second_smart_algorithm = self.evaluate_model(train_dataloader_for_evaluation, val_dataloader, apply_first_smart_algorithm=False, apply_second_smart_algorithm=True)

            self.train_accuracy_without_smart_algorithms_history.append(train_accuracy_not_smart)
            self.train_accuracy_with_first_smart_algorithm_history.append(train_accuracy_with_first_smart_algorithm)
            self.train_accuracy_with_second_smart_algorithm_history.append(train_accuracy_with_second_smart_algorithm)
            self.val_accuracy_without_smart_algorithms_history.append(val_accuracy_not_smart)
            self.val_accuracy_with_first_smart_algorithm_history.append(val_accuracy_with_first_smart_algorithm)
            self.val_accuracy_with_second_smart_algorithm_history.append(val_accuracy_with_second_smart_algorithm)

            
            # Save the best model
            if max(val_accuracy_not_smart, val_accuracy_with_first_smart_algorithm, val_accuracy_with_second_smart_algorithm) > best_model_val_accuracy:
                best_model_val_accuracy = max(val_accuracy_not_smart, val_accuracy_with_first_smart_algorithm, val_accuracy_with_second_smart_algorithm)
                torch.save(self.state_dict(), best_model_filename)

            # Step the scheduler if needed 
            if do_learning_rate_decay:
                scheduler.step()
            
            # Print the loss and accuracy of the model on the train and val datasets
            print(f"Epoch: {epoch + 1}, Training Loss: {train_loss} Val Loss: {val_loss}")
            print(f"Model accuracy on the train dataset without smart algorithms is: {train_accuracy_not_smart}")
            print(f"Model accuracy on the train dataset with first smart algorithm: {train_accuracy_with_first_smart_algorithm}")
            print(f"Model accuracy on the train dataset with second smart algorithm: {train_accuracy_with_second_smart_algorithm}")
            print("")
            print(f"Model accuracy on the val dataset without smart algorithms is: {val_accuracy_not_smart}")
            print(f"Model accuracy on the val dataset with first smart algorithm: {val_accuracy_with_first_smart_algorithm}")
            print(f"Model accuracy on the val dataset with second smart algorithm: {val_accuracy_with_second_smart_algorithm}")
            print("")

        self.plot_loss_and_accuracy_history()

    def compute_model_loss_on_dataset(self, dataloader):
        """
        Compute the model loss on the dataset

        :return: model loss on the dataset
        """
        self.eval()
        loss = 0.0
        with torch.no_grad():
            for inputs, _, labels in dataloader:
                inputs = inputs.to(config.MEMORY_DEVICE)
                labels = labels.to(config.MEMORY_DEVICE)
                y_pred_logits = self(inputs)
                labels = labels.long()
                loss += self.loss_function(y_pred_logits, labels)
            loss /= len(dataloader)
        return loss

    def get_labels_of_dataset(self, dataloader):
        """
        Get the labels of the dataset

        :return: labels of the dataset
        :rtype: torch.Tensor
        """
        self.eval()
        labels = []
        with torch.no_grad():
            for _, _, mini_batch_labels in dataloader:
                mini_batch_labels = mini_batch_labels.to(config.MEMORY_DEVICE)
                for label in mini_batch_labels:
                    labels.append(label)
        return torch.stack(labels)
    
    def get_predictions_and_labels_of_dataset(self, dataloader, apply_first_smart_algorithm=False, apply_second_smart_algorithm=True):
        """
        Get the predictions of the model on the dataset and labels of the dataset

        :return: labels and predictions of the model on the dataset
        :r1type: list
        :r2type: torch.Tensor
        """
        
        self.eval()
        predictions_probs = []
        labels = []
        
        # Get the predictions probabilities and labels
        with torch.no_grad():
            for mini_batch_inputs, _, mini_batch_labels in dataloader:
                mini_batch_inputs = mini_batch_inputs.to(config.MEMORY_DEVICE)
                mini_batch_labels = mini_batch_labels.to(config.MEMORY_DEVICE)
                mini_batch_predictions_probs = self.predict_probs(mini_batch_inputs)
                for label, prediction_prob in zip(mini_batch_labels, mini_batch_predictions_probs):
                    predictions_probs.append(prediction_prob)
                    labels.append(label)
        
        # Get the predictions
        predictions = self.get_predictions_according_predictions_probs(dataloader, predictions_probs, apply_first_smart_algorithm, apply_second_smart_algorithm)

        return labels, predictions

    def get_predictions_according_predictions_probs(self, dataloader, predictions_probs, apply_first_smart_algorithm=False, apply_second_smart_algorithm=True):
        """
        Get the predictions of the model on the dataset according to the predictions probabilities

        :return: predictions of the model on the dataset
        :rtype: torch.Tensor
        """
        predictions = []
        word_idx = 0
        words_lens = dataloader.dataset.words_lens
        curr_idx = 0

        if not apply_first_smart_algorithm and not apply_second_smart_algorithm:
            return torch.argmax(torch.stack(predictions_probs), dim=1)
            
        for word_len in words_lens:
            word_idx += 1
            word_probs = predictions_probs[curr_idx:curr_idx + word_len]
            word_predictions = torch.argmax(torch.stack(word_probs), dim=1)
            word_predictions_np = word_predictions.cpu().numpy()
            counter = Counter(word_predictions_np)
            max_count = torch.tensor(counter.most_common(1)[0][1])
            if apply_second_smart_algorithm:
                most_common_predictions = [k for k, v in counter.items() if v == max_count]
                most_common_predictions_probs = {}
                for i in range(len(word_probs)):
                    if word_predictions_np[i] in most_common_predictions:
                        most_common_predictions_probs[word_predictions_np[i]] = most_common_predictions_probs.get(word_predictions_np[i], 0) + word_probs[i][word_predictions_np[i]]
                final_prediction = torch.tensor(max(most_common_predictions_probs, key=most_common_predictions_probs.get))
            else:
                final_prediction = torch.tensor(counter.most_common(1)[0][0])
            for i in range(word_len):
                predictions.append(final_prediction)
            curr_idx += word_len

        predictions = torch.stack(predictions)    
        return predictions

    def plot_loss_and_accuracy_history(self):
        """
        Plot the loss and accuracy history of the model:
        1. Plot the loss history of the model on the train and val dataset 
        2. Plot the accuracy history of the model on the train and val dataset without smart algorithms
        3. Plot the accuracy history of the model on the train and val dataset with the first smart algorithm
        4. Plot the accuracy history of the model on the train and val dataset with the second smart algorithm
        """
        
        # Plot loss history
        self.plot_loss_history(self.train_loss_history, self.val_loss_history, "Training and Val loss history", None)

        
        # Plot Accuracy historys
        self.plot_accuracy_history(self.train_accuracy_without_smart_algorithms_history, self.val_accuracy_without_smart_algorithms_history, "Training and Val Accuracy history without smart algorithms", None)
        self.plot_accuracy_history(self.train_accuracy_with_first_smart_algorithm_history, self.val_accuracy_with_first_smart_algorithm_history, "Training and Val Accuracy history with first smart algorithm", None)
        self.plot_accuracy_history(self.train_accuracy_with_second_smart_algorithm_history, self.val_accuracy_with_second_smart_algorithm_history, "Training and Val Accuracy history with second smart algorithm", None)

    def plot_loss_history(self, train_loss_history, val_loss_history, plotting_title, axis=None):
        """
        Plot the loss history of the model on the train and val dataset

        :param train_loss_history: training loss history of the model
        :type train_loss_history: list of torch.Tensor
        :param val_loss_history: validation loss history of the model
        :type val_loss_history: list of torch.Tensor
        :param plotting_title: title of the plot
        :type plotting_title: str
        :param axis: axis to plot on, default is None

        :return: None
        """
        # Get the loss history of the model on the train and val dataset
        train_loss_history_cpu = [loss.cpu().detach().numpy() for loss in train_loss_history]
        val_loss_history_cpu = [loss.cpu().detach().numpy() for loss in val_loss_history]

        # Check if we have an axis to plot on
        if axis is None:
            fig, axis = plt.subplots(figsize=(12, 4))

        # Training and Val loss history
        axis.plot(train_loss_history_cpu, label="Training Loss")
        axis.plot(val_loss_history_cpu, label="Val Loss")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_title(plotting_title)
        axis.legend(loc="upper right")

        # Show the plot if we don't have an axis to plot on
        if axis is None:
            axis.show()

    def plot_accuracy_history(self, train_accuracy_history, val_accuracy_history, plotting_title, axis=None):
        """
        Plot the accuracy history of the model on the train and val dataset

        :param train_accuracy_history: training accuracy history of the model
        :type train_accuracy_history: list of torch.Tensor
        :param val_accuracy_history: validation accuracy history of the model
        :type val_accuracy_history: list of torch.Tensor
        :param plotting_title: title of the plot
        :type plotting_title: str
        :param axis: axis to plot on, default is None

        :return: None
        """
        # Get the accuracy history of the model on the train and val dataset
        train_accuracy_history_cpu = train_accuracy_history
        val_accuracy_history_cpu = val_accuracy_history

        # Check if we have an axis to plot on
        if axis is None:
            fig, axis = plt.subplots(figsize=(12, 4))

        # Training and Val loss history
        axis.plot(train_accuracy_history_cpu, label="Training Accuracy")
        axis.plot(val_accuracy_history_cpu, label="Val Accuracy")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Accuracy")
        axis.set_title(plotting_title)
        axis.legend(loc="upper right")

        # Show the plot if we don't have an axis to plot on
        if axis is None:
            axis.show()

    def plot_evaluation_graphs(self, dataloader, apply_first_smart_algorithm=False, apply_second_smart_algorithm=True):
        """
        Plot evaluation graphs of the model on the dataset:
        1. Plot the ROC curve of the model on the dataset
        2. Plot the Precision Recall curve of the model on the dataset
        3. Plot the confusion matrix of the model on the dataset

        :param dataloader: dataloader
        :type dataloader: torch.utils.data.DataLoader
        :param apply_first_smart_algorithm: apply the first smart algorithm, default is False
        :type apply_first_smart_algorithm: bool
        :param apply_second_smart_algorithm: apply the second smart algorithm, default is True
        :type apply_second_smart_algorithm: bool

        :return: None
        """
        # Get the labels and predictions probabilities of the model on the dataset
        self.eval()
        labels = self.get_labels_of_dataset(dataloader)
        predictions_probs = self.get_predictions_probs_of_dataset(dataloader)
        predictions = self.get_predictions_of_dataset(dataloader, apply_first_smart_algorithm, apply_second_smart_algorithm)
        labels_np = labels.cpu().numpy()
        predictions_probs_np = predictions_probs.cpu().numpy()
        predictions_np = predictions.cpu().numpy()

        # Plot evaluation graphs
        self.plot_ROC_curve(labels_np, predictions_probs_np)
        self.plot_Precision_Recall_curve(labels_np, predictions_probs_np)
        self.plot_confusion_matrix(labels_np, predictions_np)

    def plot_ROC_curve(self, labels_np, predictions_probs_np, axis=None):
        """
        Plot the ROC curve of the model on the dataset

        :param labels_np: labels of the dataset
        :type labels_np: numpy.ndarray
        :param predictions_probs_np: predictions probabilities of the model on the dataset
        :type predictions_probs_np: numpy.ndarray
        :param axis: axis to plot on, default is None

        :return: None
        """
        # Plot ROC curve
        # Check if we have an axis to plot on
        if axis is None:
            fig, axis = plt.subplots(figsize=(12, 4))
        
        for i in range(config.NUM_OF_CLASSES_IN_FONT_DATASET):
            fpr, tpr, _ = roc_curve(labels_np, predictions_probs_np[:, i], pos_label=i)
            axis.plot(fpr, tpr, label=f"Class {i}")
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
        axis.set_title("ROC Curve")
        axis.legend(loc="lower right")

        # Show the plot if we don't have an axis to plot on
        if axis is None:
            plt.show()

    def plot_Precision_Recall_curve(self, labels_np, predictions_probs_np, axis=None):
        """
        Plot the Precision Recall curve of the model on the dataset

        :param labels_np: labels of the dataset
        :type labels_np: numpy.ndarray
        :param predictions_probs_np: predictions probabilities of the model on the dataset
        :type predictions_probs_np: numpy.ndarray
        :param axis: axis to plot on, default is None

        :return: None
        """
        # Plot Precision Recall curve
        # Check if we have an axis to plot on
        if axis is None:
            fig, axis = plt.subplots(figsize=(12, 4))
        for i in range(config.NUM_OF_CLASSES_IN_FONT_DATASET):
            precision, recall, _ = precision_recall_curve(labels_np, predictions_probs_np[:, i], pos_label=i)
            axis.plot(recall, precision, label=f"Class {i}")
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_title("Precision Recall Curve")
        axis.legend(loc="lower right")

        # Show the plot if we don't have an axis to plot on
        if axis is None:
            plt.show()

    def plot_confusion_matrix(self, labels_np, predictions_np, axis=None):
        """
        Plot the confusion matrix of the model on the dataset

        :param labels_np: labels of the dataset
        :type labels_np: numpy.ndarray
        :param predictions_np: predictions of the model on the dataset
        :type predictions_np: numpy.ndarray
        :param axis: axis to plot on, default is None

        :return: None
        """
        # Plot confusion matrix
        # Check if we have an axis to plot on
        if axis is None:
            fig, axis = plt.subplots(figsize=(12, 4))
        confusion_matrix = metrics.confusion_matrix(labels_np, predictions_np)
        sns.heatmap(confusion_matrix, annot=True, fmt='g')
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_title("Confusion Matrix")
        
        # Show the plot if we don't have an axis to plot on
        if axis is None:
            plt.show()
    
    def predict_probs(self, inputs):
        """
        Predict probabilities of the model on the input data

        :param inputs: input data
        :type inputs: torch.Tensor

        :return: model predictions probabilities on the input data
        """
        self.eval()
        with torch.no_grad():
            inputs = inputs.to(config.MEMORY_DEVICE)
            y_pred_logits = self(inputs)
            return nn.functional.softmax(y_pred_logits, dim=1)

    def get_predictions_of_dataset(self, dataloader, apply_first_smart_algorithm=False, apply_second_smart_algorithm=True):
        """
        Get the predictions of the model on the dataset

        :return: predictions of the model on the dataset
        :rtype: torch.Tensor
        """
        _, predictions = self.get_predictions_and_labels_of_dataset(dataloader, apply_first_smart_algorithm, apply_second_smart_algorithm)
        return predictions
    
    def get_predictions_probs_of_dataset(self, dataloader):
        """
        Get the predictions probabilities of the model on the dataset
        
        :return: predictions probabilities of the model on the dataset
        :rtype: torch.Tensor
        """
        self.eval()
        predictions_probs = []
        with torch.no_grad():
            for mini_batch_inputs, _, _ in dataloader:
                mini_batch_inputs = mini_batch_inputs.to(config.MEMORY_DEVICE)
                mini_batch_predictions_probs = self.predict_probs(mini_batch_inputs)
                for prediction_prob in mini_batch_predictions_probs:
                    predictions_probs.append(prediction_prob)
        return torch.stack(predictions_probs)

    def show_all_the_results(self, train_dataloader_for_evaluation, val_dataloader):
        """
        Show all the results of the model on the train and val datasets

        :return: None
        """
        print("Amount of parameters of the model is:", self.calc_amount_of_parameters())
        print("")
        model_accuracy_on_train_dataset_without_smart_algorithms = self.evaluate_model_on_dataset(train_dataloader_for_evaluation, apply_first_smart_algorithm=False, apply_second_smart_algorithm=False)
        model_accuracy_on_train_dataset_with_first_smart_algorithm = self.evaluate_model_on_dataset(train_dataloader_for_evaluation, apply_first_smart_algorithm=True, apply_second_smart_algorithm=False)
        model_accuracy_on_train_dataset_with_second_smart_algorithm = self.evaluate_model_on_dataset(train_dataloader_for_evaluation, apply_first_smart_algorithm=False, apply_second_smart_algorithm=True)
        if val_dataloader:
            model_accuracy_on_val_dataset_without_smart_algorithms = self.evaluate_model_on_dataset(val_dataloader, apply_first_smart_algorithm=False, apply_second_smart_algorithm=False)
            model_accuracy_on_val_dataset_with_first_smart_algorithm = self.evaluate_model_on_dataset(val_dataloader, apply_first_smart_algorithm=True, apply_second_smart_algorithm=False)
            model_accuracy_on_val_dataset_with_second_smart_algorithm = self.evaluate_model_on_dataset(val_dataloader, apply_first_smart_algorithm=False, apply_second_smart_algorithm=True)

        print(f"Model accuracy on the train dataset without smart algorithms is: {model_accuracy_on_train_dataset_without_smart_algorithms}")
        print(f"Model accuracy on the train dataset with first smart algorithm: {model_accuracy_on_train_dataset_with_first_smart_algorithm}")
        print(f"Model accuracy on the train dataset with second smart algorithm: {model_accuracy_on_train_dataset_with_second_smart_algorithm}")
        print("")
        
        if val_dataloader:
            print(f"Model accuracy on the val dataset without smart algorithms is: {model_accuracy_on_val_dataset_without_smart_algorithms}")
            print(f"Model accuracy on the val dataset with first smart algorithm: {model_accuracy_on_val_dataset_with_first_smart_algorithm}")
            print(f"Model accuracy on the val dataset with second smart algorithm: {model_accuracy_on_val_dataset_with_second_smart_algorithm}")
            print("")
        
        print("Evaluation graphs of the model on the train dataset with second smart algorithm:")
        self.plot_evaluation_graphs(train_dataloader_for_evaluation, apply_first_smart_algorithm=False, apply_second_smart_algorithm=True)
        if val_dataloader:
            print("Evaluation graphs of the model on the val dataset with second smart algorithm:")
            self.plot_evaluation_graphs(val_dataloader, apply_first_smart_algorithm=False, apply_second_smart_algorithm=True)


    def evaluate_model(self, train_dataloader, val_dataloader, apply_first_smart_algorithm=False, apply_second_smart_algorithm=True):
        """
        Evaluate the model on the train and val dataset

        :return: accuracy of the model on the train and val dataset
        """
        # Make predictions on train and val dataset
        train_labels, train_predictions = self.get_predictions_and_labels_of_dataset(train_dataloader, apply_first_smart_algorithm=apply_first_smart_algorithm, apply_second_smart_algorithm=apply_second_smart_algorithm)
        val_labels, val_predictions = self.get_predictions_and_labels_of_dataset(val_dataloader, apply_first_smart_algorithm=apply_first_smart_algorithm, apply_second_smart_algorithm=apply_second_smart_algorithm)
        
        # Calculate and print accuracy of train and val dataset
        model_accuracy_on_train_dataset = self.calculate_accuracy(train_labels, train_predictions)
        model_accuracy_on_val_dataset = self.calculate_accuracy(val_labels, val_predictions)
        return model_accuracy_on_train_dataset, model_accuracy_on_val_dataset

    def evaluate_model_on_dataset(self, dataloader, apply_first_smart_algorithm=False, apply_second_smart_algorithm=True):
        """
        Evaluate the model on the dataset

        :return: Accuracy of the model on the dataset, amount of right predictions and amount of false predictions
        """
        # Get labels and predictions of the model on the dataset
        labels, predictions = self.get_predictions_and_labels_of_dataset(dataloader, apply_first_smart_algorithm, apply_second_smart_algorithm)

        # Calculate the accuracy of the model on the dataset
        model_accuracy_on_dataset = self.calculate_accuracy(labels, predictions)
        amount_of_right_predictions = sum([float(p == l) for p, l in zip(labels, predictions)])
        amount_of_false_predictions = sum([float(p != l) for p, l in zip(labels, predictions)]) 
        return model_accuracy_on_dataset, amount_of_right_predictions, amount_of_false_predictions
    
    def create_submit_file(self, test_loader, model_params_path, submit_file_name, apply_first_smart_algorithm=False, apply_second_smart_algorithm=True):
        # Load the model parameters
        self.load_state_dict(torch.load(model_params_path, map_location=config.MEMORY_DEVICE))
        
        # Get the predictions of the model on the test dataset with use of smart predictions
        predictions = self.get_predictions_of_dataset(test_loader, apply_first_smart_algorithm, apply_second_smart_algorithm)
        
        # Convert to numpy array
        predictions = predictions.cpu().numpy()
        
        # Create a dataframe to hold the predictions.
        new_df = pd.DataFrame({'ind': range(len(predictions)), 'font': predictions})

        #Save the dataframe to a csv file.
        new_df.to_csv(submit_file_name + '.csv', index=False)

    def calc_amount_of_parameters(self):
        """
        Calculate the amount of parameters of the model

        :return: amount of parameters of the model
        """
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def calculate_accuracy(labels, predictions):
        """
        Calculate the accuracy of the model on the dataset

        :param labels: labels of the dataset
        :type labels: torch.Tensor
        :param predictions: predictions of the model on the dataset
        :type predictions: torch.Tensor

        :return: Accuracy of the model on the dataset
        """
        total_examples = len(labels)
        number_of_correct_predictions = sum([float(p == l) for p, l in zip(labels, predictions)])
        return number_of_correct_predictions / total_examples