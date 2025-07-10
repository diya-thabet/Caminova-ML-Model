# Caminova-ML-Model


Yield Prediction Project
This repository contains code for predicting yield using a Random Forest model, based on data provided in an Excel file (dataSet.xlsx). The project includes a Python script (yield_prediction.py) and a Jupyter Notebook (yield_prediction.ipynb) that perform data preprocessing, model training, evaluation, and visualization of results.
Repository Contents

yield_prediction.py: Python script containing the complete workflow for loading data, preprocessing, training a Random Forest model, evaluating it, and generating visualizations.
yield_prediction.ipynb: Jupyter Notebook with the same functionality as the Python script, organized into cells for interactive execution.
dataSet.xlsx: Excel file containing the dataset (must include a sheet named "data for camel ").
yield_predictor.pkl: (Generated after running) The saved Random Forest model.
README.md: This file, providing project overview and instructions.

Prerequisites

Python 3.7+ (for local execution)
Libraries (install via pip):pip install pandas numpy scikit-learn matplotlib joblib


Google Colab (for running in the cloud, no local installation required)
Excel File: Ensure dataSet.xlsx is available in the working directory or upload it to Colab.

Setup Instructions
Option 1: Running in Google Colab

Upload Files:
Open Google Colab (https://colab.research.google.com).
Upload yield_prediction.ipynb and dataSet.xlsx to your Colab environment using the file upload feature in the left sidebar.
Alternatively, upload yield_prediction.py and dataSet.xlsx if you prefer to run the script.


Run the Notebook:
Open yield_prediction.ipynb in Colab.
Execute the cells sequentially by clicking the "Run" button or pressing Shift + Enter.
Ensure dataSet.xlsx is in the /content/ directory (Colab's default working directory).


Run the Python Script:
If using yield_prediction.py, create a new Colab notebook and add a cell with:!python yield_prediction.py


Ensure dataSet.xlsx is uploaded to Colab.


View Outputs:
The code will print sheet names, data shapes, model performance metrics (MSE, R², accuracy), and display seven visualizations (e.g., Actual vs Predicted, Residual Plot, Feature Importance).
The trained model will be saved as yield_predictor.pkl in the Colab environment.



Option 2: Running Locally

Clone the Repository:git clone <repository-url>
cd <repository-directory>


Install Dependencies:pip install -r requirements.txt

(Create requirements.txt with: pandas, numpy, scikit-learn, matplotlib, joblib.)
Run the Python Script:python yield_prediction.py

Ensure dataSet.xlsx is in the same directory as the script.
Run the Jupyter Notebook:
Start Jupyter Notebook:jupyter notebook


Open yield_prediction.ipynb in the browser and run all cells.


View Outputs:
The script/notebook will print data details, model metrics, and display visualizations.
The model will be saved as yield_predictor.pkl in the working directory.



Code Overview
The code performs the following steps:

Loads the Excel File: Reads dataSet.xlsx, inspects sheet names, and loads the "data for camel " sheet.
Preprocesses Data:
Uses the second row as column headers.
Cleans the Age column (maps 'Y' to 0, 'O' to 1).
Converts other columns to numeric, drops unnamed columns, and fills missing values with 0.
Drops rows where yield is missing.


Trains a Random Forest Model:
Splits data into training (80%) and testing (20%) sets.
Trains a Random Forest Regressor and predicts yield.


Evaluates the Model:
Computes Mean Squared Error (MSE), R² score, and Mean Relative Accuracy.
Saves the model as yield_predictor.pkl.


Generates Visualizations:
Scatter Plot: Actual vs Predicted Yield
Residual Plot: Residuals vs Predicted
Feature Importance: Bar plot of feature contributions
Histogram: Distribution of prediction errors
Sorted Plot: Actual vs Predicted (sorted by actual yield)
Boxplot: Distribution of prediction errors
Confidence Plot: Absolute errors vs predicted yield



Usage Notes

Data Requirements: The dataSet.xlsx file must contain a sheet named "data for camel " with a yield column and other features (e.g., Age). The second row should contain column headers.
File Path: If dataSet.xlsx is in a different directory, update file_path in the code to the correct path (e.g., /content/dataSet.xlsx in Colab or ./data/dataSet.xlsx locally).
Visualizations: Ensure a display environment (e.g., Colab, Jupyter, or a local GUI) to view plots. In non-GUI environments, save plots using plt.savefig('plot_name.png').
Model Reuse: Load the saved model with joblib.load('yield_predictor.pkl') for future predictions.

Troubleshooting

File Not Found Error: Verify dataSet.xlsx is in the correct directory or uploaded to Colab.
Missing Libraries: Install required libraries using pip or ensure you're using Colab, which has them pre-installed.
Data Issues: Check that the "data for camel " sheet exists and has valid data. Inspect sample_data output to debug.
Visualization Issues: If plots don't display, ensure plt.show() is called and you're in an environment supporting matplotlib.

Contact
For questions or issues, please open an issue in the repository or contact the repository owner.
