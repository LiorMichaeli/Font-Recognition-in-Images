Name: Lior Michaeli
ID: 328364583


The project has the following structure:

data: Folder that needs to contain the train.h5 and test.h5 files. You need to go to the Google Drive link,
download these files from the data folder, and put them in this folder. 

models:
	models_predictions_on_test: Contain the submissions files
	
	models_weights: Contain the model's weights of the models that appeared in the report.
	You need to go to the Google Drive link, download the weights files from the models_weights folder, and put them in this folder.
	
	optimized_models_params: Contain the hyperparameters of the optimized models that appeared in the report.

report: 
	report: The report of the project
	figures: Contains the figures that appeared in the report

src:
	config file: Has the configurations and constants of the project

	data: Files that handle the data

	examples: Files that contain code examples for the models in the report.
	These files have functions that show all the results of these models,
	and training and optimization functions for these models.
	
	models: Files that contain the code of the models that appeared in the report.

	visualization: Files that have functions for visualization of the data and the model's predictions

	main file: Has code that loads the final model and creates for him a submission file.
	In the main file, there are Code snippets that are in comments. 
	These Code snippets show examples of the project using the example files and show data samples
	and data distribution. If you want you can run these Code snippets.

README.md: This file

requirments.txt: File that contains all the requirements of this project. To use this file do the following steps:
1. Open the terminal in the project folder
2. Write: pip install -r requirments.txt

Google Drive Link: Text file that contains a link to Google Drive, where the data and the model weights located

Note: 
In this project, I left only the code of the models described in Tables 1, 2, and 3 in the report.
For example, I didn't put in my code my little experiment with Multi Neural networks that was described in the report. 
In summary, I left only the code for production.

Note:
I trained the models on Kaggle, because I don't have a GPU and good enough hardware on my computer.