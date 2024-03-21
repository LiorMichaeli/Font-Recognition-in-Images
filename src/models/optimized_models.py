
# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import optuna


# Import files
from config import config
import models.Base_model as Base_model


class OptimizeResNetModel(Base_model.BaseModel):
    """
    This class is used to optimize the hyper-parameters of the Custom ResNet model.

    The hyper-parameters of the model that are optimized are:
    1. Number of ResNet blocks
    2. Starting number of channels
    3. Number of fully connected layers
    4. Number of neurons in the fully connected layers

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
    
    Hyper-parameters:
    - **number_of_ResNet_blocks** (int) - Number of ResNet blocks
    
    Model layers:
    - **start_block** (nn.Sequential) - Start block of the model
    - **max_pool** (nn.MaxPool2d) - Max Pooling layer
    - **fc_layers** (nn.Sequential) - Fully connected layers
    - More layers are defined according to the hyper-parameters

    Loss function:
    - **loss_function** (nn.CrossEntropyLoss) - Loss function

    """
    def __init__(self, trial, params=None):
        """
        Constructor for the OptimizeResNetModel class.

        :param trial: (optuna.trial) The trial object that is used to generate hyper-parameters
        :param params: (dict) Dictionary containing the hyper-parameters of the model if they are not optimized, default is None

        :return: None
        """
        super(OptimizeResNetModel, self).__init__()

        # Define layers

        if params:
            self.number_of_ResNet_blocks = params["number_of_ResNet_blocks"]
            starting_channels = params["starting_number_of_channels"]
            number_of_fc_layers = params["number_of_fc_layers"]
        else:
            self.number_of_ResNet_blocks = trial.suggest_int("number_of_ResNet_blocks", 1, 3)
            starting_channels = trial.suggest_categorical("starting_number_of_channels", [16, 32, 64])
            number_of_fc_layers = trial.suggest_int("number_of_fc_layers", 0, 2)
        
        # Define the conv blocks
        self.create_conv_blocks(starting_channels)

        # Define the fully connected layers
        in_features_fc1 = int((config.INPUT_SHAPE[1] / 4) * (config.INPUT_SHAPE[2] / 4) * starting_channels * pow(2, self.number_of_ResNet_blocks))
        self.create_fc_layers(in_features_fc1, number_of_fc_layers, trial, params)

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

    def create_conv_blocks(self, starting_amount_of_channels):
        """
        Create the convolutional blocks of the model.

        :param starting_amount_of_channels: (int) The starting amount of channels

        :return: None
        """
        # Convolutional layers
        in_channels = config.INPUT_SHAPE[0]
        out_channels = starting_amount_of_channels

        for i in range(self.number_of_ResNet_blocks + 1):
            if i == 0:
                self.start_block = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
            else:
                # Add a skip connection
                if i == 1:
                    setattr(self, f'conv_skip_block{i}', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1))
                else:
                    setattr(self, f'conv_skip_block{i}', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))

                # First convolutional layer
                setattr(self, f'conv1_block{i}', nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1))
                setattr(self, f'bn2d_1_block{i}', nn.BatchNorm2d(in_channels))
                setattr(self, f'relu1_block{i}', nn.ReLU())

                # Second convolutional layer
                if i == 1:
                    setattr(self, f'conv2_block{i}', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1))
                else:
                    setattr(self, f'conv2_block{i}', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
                setattr(self, f'bn2d_2_block{i}', nn.BatchNorm2d(out_channels))
                setattr(self, f'relu2_block{i}', nn.ReLU())

            in_channels = out_channels
            out_channels *= 2

        # Max Pooling layer and Flatten layer
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def create_fc_layers(self, in_features_fc1, number_of_fc_layers, trial, params=None):
        """
        Create the fully connected layers of the model.

        :param in_features_fc1: (int) The number of features that are outputted from the last convolutional layer
        :param number_of_fc_layers: (int) The number of fully connected layers
        :param trial: (optuna.trial) The trial object that is used to generate hyper-parameters
        :param params: (dict) Dictionary containing the hyper-parameters of the model if they are not optimized, default is None
        
        :return: None
        """
        fc_layers = []
        in_features = in_features_fc1
        
        # Define the fc layers
        for i in range(number_of_fc_layers):
            if params:
                out_features = params[f"fc{i + 1}_out_features"]
            else:
                out_features = trial.suggest_int(f"fc{i + 1}_out_features", 512, in_features)
            fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(0.5))
            in_features = out_features
        
        fc_layers.append(nn.Linear(in_features, config.NUM_OF_CLASSES_IN_FONT_DATASET))
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        """
        Forward pass of the model.

        :param x: (torch.Tensor) The input data

        :return: (torch.Tensor) The output of the model
        """
        # Start Block
        x = self.start_block(x)

        # ResNet blocks
        for i in range(self.number_of_ResNet_blocks):
            skip = getattr(self, f'conv_skip_block{i + 1}')(x)
            x = getattr(self, f'conv1_block{i + 1}')(x)
            x = getattr(self, f'bn2d_1_block{i + 1}')(x)
            x = getattr(self, f'relu1_block{i + 1}')(x)
            x = getattr(self, f'conv2_block{i + 1}')(x)
            x = torch.add(x, skip)
            x = getattr(self, f'bn2d_2_block{i + 1}')(x)
            x = getattr(self, f'relu2_block{i + 1}')(x)
        
        # Max Pooling
        x = self.max_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc_layers(x)
        
        return x
    
    def train_model(self, trial, train_dataloader, val_dataloader, train_dataloader_for_evaluation, best_model_filename,
                     num_epochs=10, learning_rate=0.01, lambd=0.0005,
                       do_learning_rate_decay=True, learning_rate_decay_factor=0.1, learning_rate_decay_step=15):
        # Define a optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=lambd)
        
        # Define a learning rate scheduler
        if do_learning_rate_decay:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=learning_rate_decay_step, gamma=learning_rate_decay_factor)

        best_model_val_accuracy = -float('infinity')
        best_model_train_accuracy = -float('infinity')

        for epoch in range(num_epochs):
            print(f"Epoch: {epoch + 1}")
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
            if val_dataloader:
                val_loss = self.compute_model_loss_on_dataset(val_dataloader)
                self.val_loss_history.append(val_loss)
            if val_dataloader:
                train_accuracy_not_smart, val_accuracy_not_smart = self.evaluate_model(train_dataloader, val_dataloader, apply_first_smart_algorithm=False, apply_second_smart_algorithm=False)
                train_accuracy_with_first_smart_algorithm, val_accuracy_with_first_smart_algorithm = self.evaluate_model(train_dataloader_for_evaluation, val_dataloader, apply_first_smart_algorithm=True, apply_second_smart_algorithm=False)
                train_accuracy_with_second_smart_algorithm, val_accuracy_with_second_smart_algorithm = self.evaluate_model(train_dataloader_for_evaluation, val_dataloader, apply_first_smart_algorithm=False, apply_second_smart_algorithm=True)
            else:
                train_accuracy_not_smart = self.evaluate_model_on_dataset(train_dataloader, apply_first_smart_algorithm=False, apply_second_smart_algorithm=False)[0]
                train_accuracy_with_first_smart_algorithm = self.evaluate_model_on_dataset(train_dataloader_for_evaluation, apply_first_smart_algorithm=True, apply_second_smart_algorithm=False)[0]
                train_accuracy_with_second_smart_algorithm = self.evaluate_model_on_dataset(train_dataloader_for_evaluation, apply_first_smart_algorithm=False, apply_second_smart_algorithm=True)[0]

            self.train_accuracy_without_smart_algorithms_history.append(train_accuracy_not_smart)
            self.train_accuracy_with_first_smart_algorithm_history.append(train_accuracy_with_first_smart_algorithm)
            self.train_accuracy_with_second_smart_algorithm_history.append(train_accuracy_with_second_smart_algorithm)
            if val_dataloader:
                self.val_accuracy_without_smart_algorithms_history.append(val_accuracy_not_smart)
                self.val_accuracy_with_first_smart_algorithm_history.append(val_accuracy_with_first_smart_algorithm)
                self.val_accuracy_with_second_smart_algorithm_history.append(val_accuracy_with_second_smart_algorithm)
            
            if trial is None:
                if val_dataloader:
                    print(f"Epoch: {epoch + 1}, Training Loss: {train_loss} Val Loss: {val_loss}")
                else:
                    print(f"Epoch: {epoch + 1}, Training Loss: {train_loss}")
                print(f"Model accuracy on the train dataset without smart algorithms is: {train_accuracy_not_smart}")
                print(f"Model accuracy on the train dataset with first smart algorithm: {train_accuracy_with_first_smart_algorithm}")
                print(f"Model accuracy on the train dataset with second smart algorithm: {train_accuracy_with_second_smart_algorithm}")
                print("")
                if val_dataloader:
                    print(f"Model accuracy on the val dataset without smart algorithms is: {val_accuracy_not_smart}")
                    print(f"Model accuracy on the val dataset with first smart algorithm: {val_accuracy_with_first_smart_algorithm}")
                    print(f"Model accuracy on the val dataset with second smart algorithm: {val_accuracy_with_second_smart_algorithm}")
                    print("")
            
            # Save the best model
            if val_dataloader:
                if max(val_accuracy_with_second_smart_algorithm, val_accuracy_with_first_smart_algorithm, val_accuracy_not_smart) > best_model_val_accuracy:
                    best_model_val_accuracy = max(val_accuracy_with_second_smart_algorithm, val_accuracy_with_first_smart_algorithm, val_accuracy_not_smart)
                    torch.save(self.state_dict(), best_model_filename)
            else:
                if max(train_accuracy_with_second_smart_algorithm, train_accuracy_with_first_smart_algorithm, train_accuracy_not_smart) > best_model_train_accuracy:
                    best_model_train_accuracy = train_accuracy_with_second_smart_algorithm
                    torch.save(self.state_dict(), best_model_filename)

            # Step the scheduler if needed 
            if do_learning_rate_decay:
                scheduler.step()

            # Optuna - Report the current value of the objective function
            if trial:
                trial.report(best_model_val_accuracy, epoch)

            # Handle pruning based on the current value of the objective function
            if trial and trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        if trial is None:
            self.plot_loss_and_accuracy_history()


        if val_dataloader:
            return best_model_val_accuracy
        else:
            return best_model_train_accuracy
            

class Optimization_Handler():
    """
    This class is used to optimize the hyper-parameters of the model.
    This class uses the Optuna library to optimize the hyper-parameters and handles the optimization process.

    :Attributes:
    - **model** (list) - The model that is optimized
    - **train_loader** (torch.utils.data.DataLoader) - The train loader
    - **val_loader** (torch.utils.data.DataLoader) - The validation loader
    - **train_loader_for_evaluation** (torch.utils.data.DataLoader) - The train loader for evaluation
    - **best_models_folder_path** (str) - The folder path where the best models are saved
    - **study** (optuna.study) - The study object

    :Methods:
    - **objective(trial)** - The objective function that is optimized
    - **optimize_model()** - Optimize the model
    - **show_results_of_spesific_trial(trial, trial_model_path)** - Show the results of a specific trial
    """
    def __init__(self, train_loader, val_loader, train_loader_for_evaluation, best_models_path):
        """
        Constructor for the Optimization_Handler class.

        :param train_loader: (torch.utils.data.DataLoader) The train loader
        :param val_loader: (torch.utils.data.DataLoader) The validation loader
        :param train_loader_for_evaluation: (torch.utils.data.DataLoader) The train loader for evaluation
        :param best_models_path: (str) The path where the best models are saved

        :return: None
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_loader_for_evaluation = train_loader_for_evaluation
        self.best_models_path = best_models_path
        self.study = None

    def objective(self, trial):
        """
        The objective function for the optimization process.
        This function is used to optimize the hyper-parameters of the model and the hyper-parameters of the training process:
        1. Hyper-parameters of the model
        2. Learning rate
        3. Weight decay

        :param trial: (optuna.trial) The trial object that is used to generate hyper-parameters

        :return: (float) The value of the objective function
        """
        
        # Optuna - Create the model
        model = OptimizeResNetModel(trial)
        model = model.to(config.MEMORY_DEVICE)
        
        # Optuna - Suggestions for hyper-parameters of the training process
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        # Optuna - Define the best model filename
        if not hasattr(Optimization_Handler.objective, "counter"):
            Optimization_Handler.objective.counter = 1
        else:
            Optimization_Handler.objective.counter += 1
        best_model_filename = f"{self.best_models_path}_best_model_{Optimization_Handler.objective.counter}.pth"

        # Optuna - Train the model
        best_model_val_accuracy = model.train_model(trial, self.train_loader, self.val_loader, self.train_loader_for_evaluation, best_model_filename, num_epochs=30, learning_rate=lr, lambd=weight_decay, do_learning_rate_decay=True, learning_rate_decay_factor=0.1, learning_rate_decay_step=15)

        return best_model_val_accuracy
    
    def objective_final(self, trial):
        optimized_model_path = f"../models/models_weights/Optimized_Custom_ResNet_Table2_1.pth"
        optimized_model = OptimizeResNetModel(trial, params={
            "number_of_ResNet_blocks": 3,
            "starting_number_of_channels": 16,
            "number_of_fc_layers": 1,
            "fc1_out_features": 3297,
        })
        optimized_model = optimized_model.to(config.MEMORY_DEVICE)
        optimized_model.load_state_dict(torch.load(optimized_model_path, map_location=config.MEMORY_DEVICE))
        
        # Optuna - Suggestions for hyper-parameters of the training process
        lr = trial.suggest_float("lr", 5e-6, 5e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        num_epochs = trial.suggest_int("num_epochs", 2, 10)

        # Optuna - Define the best model filename
        if not hasattr(Optimization_Handler.objective_final, "counter"):
            Optimization_Handler.objective_final.counter = 1
        else:
            Optimization_Handler.objective_final.counter += 1
        best_model_filename = f"{self.best_models_path}_best_model_{Optimization_Handler.objective.counter}.pth"        

        # Optuna - Train the model
        best_model_val_accuracy = optimized_model.train_model(trial, self.train_loader, self.val_loader, self.train_loader_for_evaluation, best_model_filename, num_epochs=num_epochs, learning_rate=lr, lambd=weight_decay, do_learning_rate_decay=False)

        return best_model_val_accuracy

    def optimize_model(self, number_of_trials, final_optimization=True):
        """
        Optimize the model with the Optuna library.

        :param number_of_trials: (int) The number of trials

        :return: None
        """
        # Optuna - Create a study
        self.study = optuna.create_study(direction="maximize")
            
        # Optuna - Optimize the model
        if final_optimization:
            self.study.optimize(self.objective_final, n_trials=number_of_trials)
        else:
            self.study.optimize(self.objective, n_trials=number_of_trials)

    def show_results_of_spesific_trial(self, trial_idx, trial_model_path, is_final_optimization=True):
        """
        Show the results of a specific trial.

        :param trial: (int) The index of the trial (0, 1, 2, ...)
        :param trial_model_path: (str) The path of the model that is saved in the trial
        :param is_final_optimization: (bool) True if the optimization is the final optimization, default is True

        :return: None
        """
        trial = self.study.trials[trial_idx]
        params = trial.params
        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in params.items():
            print("    {}: {}".format(key, value))

        if is_final_optimization:
            optimized_model = OptimizeResNetModel(None, params = {
            "number_of_ResNet_blocks": 3,
            "starting_number_of_channels": 16,
            "number_of_fc_layers": 1,
            "fc1_out_features": 3297,
        })
        else:
            optimized_model = OptimizeResNetModel(None, params)
        optimized_model = optimized_model.to(config.MEMORY_DEVICE)
        optimized_model.load_state_dict(torch.load(trial_model_path, map_location=config.MEMORY_DEVICE))

        if self.train_loader_for_evaluation and self.val_loader:
            model_accuracy_on_train_dataset_without_smart_algos, model_accuracy_on_val_dataset_without_smart_algos = optimized_model.evaluate_model(self.train_loader_for_evaluation, self.val_loader , apply_first_smart_algorithm=False, apply_second_smart_algorithm=False)
            model_accuracy_on_train_dataset_with_first_smart_algos, model_accuracy_on_val_dataset_with_first_smart_algos = optimized_model.evaluate_model(self.train_loader_for_evaluation, self.val_loader , apply_first_smart_algorithm=True, apply_second_smart_algorithm=False)
            model_accuracy_on_train_dataset_with_second_smart_algos, model_accuracy_on_val_dataset_with_second_smart_algos = optimized_model.evaluate_model(self.train_loader_for_evaluation, self.val_loader , apply_first_smart_algorithm=False, apply_second_smart_algorithm=True)
            
            print("Model accuracy on the train dataset without smart algorithms is: ", model_accuracy_on_train_dataset_without_smart_algos)
            print("Model accuracy on the train dataset with first smart algorithm: ", model_accuracy_on_train_dataset_with_first_smart_algos)
            print("Model accuracy on the train dataset with second smart algorithm: ", model_accuracy_on_train_dataset_with_second_smart_algos)
            print("")
            print("Model accuracy on the val dataset without smart algorithms is: ", model_accuracy_on_val_dataset_without_smart_algos)
            print("Model accuracy on the val dataset with first smart algorithm: ", model_accuracy_on_val_dataset_with_first_smart_algos)
            print("Model accuracy on the val dataset with second smart algorithm: ", model_accuracy_on_val_dataset_with_second_smart_algos)
        
        elif self.train_loader_for_evaluation:
            model_accuracy_on_train_dataset_without_smart_algos = optimized_model.evaluate_model_on_dataset(self.train_loader_for_evaluation, apply_first_smart_algorithm=False, apply_second_smart_algorithm=False)
            model_accuracy_on_train_dataset_with_first_smart_algos = optimized_model.evaluate_model_on_dataset(self.train_loader_for_evaluation, apply_first_smart_algorithm=True, apply_second_smart_algorithm=False)
            model_accuracy_on_train_dataset_with_second_smart_algos = optimized_model.evaluate_model_on_dataset(self.train_loader_for_evaluation, apply_first_smart_algorithm=False, apply_second_smart_algorithm=True)

            print("Model accuracy on the train dataset without smart algorithms is: ", model_accuracy_on_train_dataset_without_smart_algos)
            print("Model accuracy on the train dataset with first smart algorithm: ", model_accuracy_on_train_dataset_with_first_smart_algos)
            print("Model accuracy on the train dataset with second smart algorithm: ", model_accuracy_on_train_dataset_with_second_smart_algos)