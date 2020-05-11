# Concrete-Compressiveness-Supervised-Learning-Comparison-Study

This is a project that runs a comparison stuy on three supervised learning methods for the dataset, a written report details the results. 

To run this code, the following packages need to be instaled: 
- time
- sklearn
- pandas
- itertools
- keras
- seaborn
- xgboost

The files project.py and krr.py contain all code for this project. krr.py contains code for kernel ridge regression and project.py contains code for everything else. Run them in python 3 with the commands: 
python project.py
python krr.py

Both python scripts are well documented, with comment blocks introducing each class and it's purpose. The main methods of both contain code to run the functions that would duplicate my results. Uncomment what is needed and run to duplicate the results. 

The dataset is located in Concrete_Data.csv. No changes are needed to this file. It is left unchanged from the downloadable dataset online with the exception that I deleted the first row that contained the names of each input or output just to keep it simple. The names are in a chart in my written report and also available online at http://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength 

The csv file NN_trained.csv contains raw output data from the neural net regressor. This should not be needed but I've included it just in case anyone duplicating this project wants to save time by not running the neural net over again. 

In case the files get mixed up: The code is designed to be in the same folder as the dataset. So krr.py, project.py, and Concrete_Data.csv should all be located in the same folder.
