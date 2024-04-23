README file for CS498 Price Prediction Software

This file's purpose is to predict the price of an AirBnB per night
utilizing real AirBnB data and various machine learning models.


TO RUN ALL FILES AT ONCE:
1. Just run main.py. 


TO RUN FILES INDIVIDUALLY:
1. Ensure all of the directories contain their proper files and all files are included. These include all of the python files listed below, and the two csv files found in the data directory. 

2. Run visualize_data.py. The respective graphs should be placed in the visualized_dataset directory. 

3. Run preprocessor.py. The respective output csv's should be placed in the prep_data directory.

4. Run model_train.py. This file will train the model based on the preprocessed training file, and create a saved model in your working directory as well as graphs/metrics in the training_metrics directory. 

5. Run model_test.py. This will take the preprocessed testing file and predict the prices, outputting the resulting csv and a graph to the prediction_metrics directory. 


DIRECTORY DESCRIPTION
- data
    - Contains two datasets, both of which contain real AirBnB price data from https://www.kaggle.com/datasets/rudymizrahi/airbnb-listings-in-major-us-cities-deloitte-ml

- prediction_metrics
    - Holds the price distribution plot and the csv containing the predicted prices after running model_predict.py. 

- prep_data
    - Contains the preprocessed csv files for training and testing after running preprocessor.py

- training_metrics
    - Contains all of the saved graphs and metrics from running model_train.py

- visualized_dataset
    - Contains all of the files that visualize the dataset after running visualize_data.py


Final Tested Metrics:
Test set evaluation:
MAE: 32.78825718758406
RMSE: 48.459525900397075
R2 Score: 0.6138027147293142
Validation set evaluation:
MAE: 33.500303929552125
RMSE: 50.18023828600664
R2 Score: 0.60057785237698

This equates to the program being able to predict the price within about $30
of the correct price, and explain the variance of the data about 60% of the time. 
These metrics are acceptable but not great, providing a very solid background for improvement
given more computing power and time. 

