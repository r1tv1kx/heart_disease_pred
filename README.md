# Heart Disease Prediction

## Overview

This project focuses on building a machine learning model to predict the likelihood of heart disease based on various clinical factors. The dataset used contains information about patients, including their age, sex, cholesterol levels, blood pressure, and other relevant measurements. This project includes exploratory data analysis (EDA), data preprocessing, model building (using Logistic Regression and Random Forest), and model evaluation.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Dataset Description](#dataset-description)
3.  [Project Structure](#project-structure)
4.  [Dependencies](#dependencies)
5.  [Setup](#setup)
6.  [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
7.  [Data Preprocessing](#data-preprocessing)
8.  [Model Building](#model-building)
    *   [Logistic Regression](#logistic-regression)
    *   [Random Forest](#random-forest)
9.  [Hyperparameter Tuning](#hyperparameter-tuning)
10. [Model Evaluation](#model-evaluation)
11. [Results](#results)
12. [Future Work](#future-work)
13. [Contributing](#contributing)

## 1. Introduction

Heart disease is a leading cause of death worldwide, making early detection and prediction crucial. This project aims to develop a predictive model that can assist healthcare professionals in identifying individuals at high risk of heart disease, enabling timely intervention and treatment.

## 2. Dataset Description

The dataset used in this project is the "Heart Disease" dataset, which includes the following features:

*   `age`: Age of the patient
*   `sex`: Sex of the patient (1 = male; 0 = female)
*   `cp`: Chest pain type (0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic)
*   `trestbps`: Resting blood pressure (in mm Hg on admission to the hospital)
*   `chol`: Serum cholesterol in mg/dl
*   `fbs`: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
*   `restecg`: Resting electrocardiographic results (0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy)
*   `thalach`: Maximum heart rate achieved
*   `exang`: Exercise-induced angina (1 = yes; 0 = no)
*   `oldpeak`: ST depression induced by exercise relative to rest
*   `slope`: The slope of the peak exercise ST segment (0 = Upsloping, 1 = Flat, 2 = Downsloping)
*   `ca`: Number of major vessels (0-3) colored by fluoroscopy
*   `thal`: Thalassemia (0 = null, 1 = fixed defect, 2 = normal blood flow, 3 = reversible defect)
*   `target`: Presence of heart disease (1 = yes; 0 = no)

The dataset has 303 rows and 14 columns.

## 3. Project Structure

<img width="566" alt="Screenshot 2025-03-01 at 11 23 11 AM" src="https://github.com/user-attachments/assets/8a080a85-d79e-40ad-862a-7f7a962bba15" />


## 4. Dependencies

The following libraries are required to run this project:

*   `python` (>=3.6)
*   `numpy`
*   `pandas`
*   `scikit-learn`
*   `matplotlib`
*   `seaborn`

You can install these dependencies using `pip`:


## 5. Setup

1.  Clone the repository:

    ```
    git clone https://github.com/r1tv1kx/heart_disease_pred.git
    cd heart_disease_pred
    ```

2.  Create a virtual environment (optional but recommended):

    ```
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  Install the dependencies:

    ```
    pip install -r requirements.txt # You might need to create this file, so follow step 4
    ```

4. Create a `requirements.txt` file

    ```
    pip freeze > requirements.txt
    ```

5.  Open and run the Jupyter Notebook:

    ```
    jupyter notebook Heart-Disease-Prediction.ipynb
    ```

## 6. Exploratory Data Analysis (EDA)

The EDA process involves exploring the dataset to understand its characteristics, identify patterns, and gain insights. Key steps include:

*   **Descriptive Statistics:** Calculating summary statistics (mean, median, std, etc.) for numerical features.
*   **Histograms:** Visualizing the distribution of individual features to understand data spread and identify skewness.
*   **Correlation Matrix:** Generating a heatmap of the correlation matrix to understand the relationships between different features and identify potential multicollinearity.
*   **Target Variable Analysis:** Displaying the distribution of the target variable to understand if the dataset is balanced or imbalanced.

## 7. Data Preprocessing

Data preprocessing is performed to prepare the data for model building. The following steps are included:

*   **Feature Scaling:** Scaling numerical features using `StandardScaler` to have zero mean and unit variance. This is important for algorithms like SVM and k-NN.
*   **One-Hot Encoding:** Converting categorical variables into numerical format using one-hot encoding. The `drop_first=True` argument is used to avoid multicollinearity.

## 8. Model Building

### Logistic Regression

A Logistic Regression model is built and trained to predict the likelihood of heart disease. Key steps include:

*   Splitting the data into training and testing sets.
*   Training the Logistic Regression model using the training data.
*   Evaluating the model's performance using accuracy, classification report, and confusion matrix.
*   Calculating and plotting the AUC-ROC curve to measure the model's ability to distinguish between positive and negative classes.

### Random Forest

A Random Forest model is also built and trained. Key steps include:

*   Training the Random Forest model using the training data.
*   Evaluating the model's performance using accuracy, classification report, and confusion matrix.
*   Extracting and displaying feature importances from the Random Forest model to understand the relative contribution of each feature in the model's predictions.

## 9. Hyperparameter Tuning

Hyperparameter tuning is performed using `GridSearchCV` to find the optimal hyperparameters for the Random Forest model. The following hyperparameters are tuned:

*   `n_estimators`: Number of trees in the forest
*   `max_depth`: Maximum depth of the trees
*   `min_samples_split`: Minimum number of samples required to split an internal node

## 10. Model Evaluation

The performance of the models is evaluated using the following metrics:

*   **Accuracy:** The proportion of correctly classified instances.
*   **Precision:** The ratio of true positives to the total number of instances classified as positive.
*   **Recall:** The ratio of true positives to the total number of actual positive instances.
*   **F1-Score:** The harmonic mean of precision and recall.
*   **AUC-ROC:** The area under the Receiver Operating Characteristic curve, which measures the model's ability to distinguish between positive and negative classes.

## 11. Results

The results of the model building and evaluation are summarized below:

*   **Logistic Regression:**
    *   Accuracy: \[Insert Accuracy Value]
    *   AUC-ROC: \[Insert AUC-ROC Value]
*   **Random Forest:**
    *   Accuracy: \[Insert Accuracy Value]
*   **Random Forest with Hyperparameter Tuning:**
    *   Best Parameters: \[Insert Best Parameters]
    *   Test Accuracy with Best Model: \[Insert Accuracy Value]

## 12. Future Work

Potential areas for future work include:

*   Experimenting with different machine learning models (e.g., SVM, Gradient Boosting).
*   Exploring feature engineering techniques to create new features from existing ones.
*   Collecting more data to improve model performance.
*   Deploying the model as a web application or API for real-time predictions.

## 13. Contributing

Contributions to this project are welcome. To contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with descriptive commit messages.
4.  Submit a pull request.



