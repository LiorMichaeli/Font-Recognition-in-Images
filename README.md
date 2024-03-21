<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/LiorMichaeli/Font-Recognition-in-Images">
    <img src="Images for README/README Logo.png" alt="Logo" width="600" height="300">
  </a>
  <h3 align="center">Font Recognition in Images</h3>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#project-structure">Project structure</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#download-data">Download data</a></li>
        <li><a href="#download-models-weights">Download model's weights</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Welcome to the font recognition in images project!

I participated in a master's degree course dealing with computer vision and deep learning.
In this course, I was working on a project dealing with the task of font recognition in images (when there was a competition between the students in the class).

Font recognition is a very important task, and it can help a lot to many people. For example, font is one of the most fundamental concepts in design, and therefore automatic font recognition from an image will help a lot to designers. As we know, Deep learning approaches are applied to many domains and tasks today, including Document Analysis tasks. In this competition, we needed to propose solutions for font recognition in images task on a special dataset, which we were provided with in the competition.

In the competition I wrote code and I wrote a technical report on the project in the format of a paper.
More detailed information appears in the technical report.

I was very enthusiastic and enjoyed the project and I hope you will be enthusiastic about it too!

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

The main libraries that I used in this project are:

* [![PyTorch][PyTorch.js]][PyTorch-url]
* [![Matplotlib][Matplotlib.js]][Matplotlib-url]
* [![Numpy][Numpy.js]][Numpy-url]
* [![scikit-learn][scikit-learn.js]][scikit-learn-url]
* [![pandas][pandas.js]][pandas-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- PROJECT STRUCTURE -->
## Project structure
The project has the following structure:

```
├───Images for README
├───data
├───models
│   ├───models_predictions_on_test
│   ├───models_weights
│   └───optimized_models_hyperparams
├───report
│   ├───report.pdf
│   └───figures
├───src
    ├───config
    ├───data
    ├───examples
    ├───models
    ├───visualization
    └───main.py
LICENSE.txt
README.md
requirments.txt
```

Here is the explanation of the role of each folder in the project:

* Images for README: Folder that contains images for README.

* data: Folder that need to contain the train.h5 and test.h5 files.
  If you need the data files go to the Download data section in Getting Started section.

* models:

  * models_predictions_on_test: Contains the submissions files, i.e., the models predictions on the test dataset.
  
  * models_weights: Folder that need to contain the model's weights of the models that appeared in the report.
	  If you need the model's weights of the models that appeared in the report, go to the Download models weights section in Getting Started section.
 
  * optimized_models_params: Contain the hyperparameters of the optimized models that appeared in the report.

* report: Folder that contains the technical report and his figures.
  * report.pdf: The technical report of the project
	* figures: Contains the figures that appeared in the technical report

* src: Folder that contains the src code files of the project
	* config file: Has the configurations and constants of the project

  * data: Files that handle the data

  * examples: Files that contain code examples for the models in the report.
	  These files have functions that show all the results of these models,
	  and training and optimization functions for these models.
	
  * models: Files that contain the code of the models that appeared in the report.

  * visualization: Files that have functions for visualization of the data and the model's predictions

  * main.py: Has code that loads the final model and creates for him a submission file.
	  In the main file, there are Code snippets that are in comments. 
	  These Code snippets show examples of the project using the example files and show data samples
	  and data distribution. If you want you can run these Code snippets.

* LICENSE.txt: The LICENSE file of the project

* README.md: The README file of the project

* requirments.txt: File that contains all the requirements of this project. To use this file do the following steps:

Note: 
In this project, I left only the code of the models described in Tables 1, 2, and 3 in the report.
For example, I didn't put in my code my little experiment with Multi Neural networks that was described in the report. 
In summary, I left only the code for production.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running my project on your computer, follow these simple steps.

### Prerequisites

Firstly, you need to have an IDE like Visual Studio Code, PyCharm, or another IDE that supports Python.

After this, you need to install all the libraries that I used in the project.
To do this, follow these simple steps:

1. Open the terminal in the project folder.
   
2. Write in the terminal:
   ```
    pip install -r requirments.txt
   ```

### Installation

This is how you can install my project on your computer:

1. Go to your terminal

2. Go to the path that you want to download to project into and write in the terminal:
   ```
   cd path_that_you_want
   ```

3. Clone the repo, i.e., write in the terminal:
   ```
   git clone https://github.com/LiorMichaeli/Font-Recognition-in-Images.git
   ```

### Download data

If you want to run and train the models yourself, you need to download the data.
If you only want to see my code and the technical report, this is not necessary.
I went about the complete process of the project in the technical report.

To download the data for the project, follow these simple steps:
1. Go to the Google Drive Link file in the data folder.
2. Go the the link in the Google Drive Link file.
3. Go to data folder and download from this folder the train.h5 and test.h5 files.
4. Put these files in the data folder(The data folder of the project, where the Google Drive Link file located).

### Download models weights

If you want to load the model's weights of the models that appeared in the report.
If you only want to see my code and the technical report, this is not necessary.
I went about the complete process of the project in the technical report.

To download the model's weights of the models that appeared in the report, follow these simple steps:
1. Go to the Google Drive Link file in the models\models_weights folder.
2. Go the the link in the Google Drive Link file.
3. Go to models_weights folder and download from this folder the model's weights that you want to load.
4. Put these files in the models\models_weights folder(The models\models_weights folder of the project, where the Google Drive Link file located).   

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

I would love to hear from you what you think about the project.

Lior Michaeli - [@LinkedIn](https://www.linkedin.com/in/liormichaeli/) - liorm0995@gmail.com

Project Link: [https://github.com/LiorMichaeli/Font-Recognition-in-Images](https://github.com/LiorMichaeli/Font-Recognition-in-Images)
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

These are the main resources that helped me in the project, and I want to give credit to them:

* [Choose an Open Source License](https://choosealicense.com)
* [Kaggle site](https://www.kaggle.com/)
* [Medium site](https://medium.com/)
* [For README](https://github.com/othneildrew/Best-README-Template/tree/master#readme-top)
* [For Markdown Badges](https://github.com/Ileriayo/markdown-badges?tab=readme-ov-file#table-of-contents)
  
And I used many more resources like sites, videos and papers. Many of them have a reference in the technical report.
  
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[PyTorch.js]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[Matplotlib.js]: https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=matplotlib&logoColor=white
[Matplotlib-url]: https://matplotlib.org/
[Numpy.js]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[Numpy-url]: https://numpy.org/
[scikit-learn.js]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[scikit-learn-url]: https://scikit-learn.org/stable/
[pandas.js]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[pandas-url]: https://pandas.pydata.org/
