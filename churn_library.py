'''
A library for predicting customer churns.

This library provides functions for importing data,
performing exploratory data analysis (EDA),
feature engineering, training models,
and generating classification reports and feature importance plots.

Functions:
- import_data(pth): Returns a pandas dataframe for the CSV file
found at the specified path.
- perform_eda(data_frame): Performs exploratory data
analysis on the dataframe and saves the resulting figures as images.
- encoder_helper(data_frame, category_lst, response): Helper
function to encode categorical columns into new columns
with the proportion of churn for each category.
- perform_feature_engineering(data_frame, response):
Performs feature engineering on the dataframe and returns
the training and testing data.
- generate_report_image(report, report_name): Converts a
classification report into an image and saves it.
- create_classification_report_image(y_train, y_test,
y_train_preds_lr, y_train_preds_rf, y_test_preds_lr,
y_test_preds_rf): Generates classification reports
for training and testing results and stores them as images.
- feature_importance_plot(model, X_data, output_pth):
Creates and stores a feature importance plot.
- train_models(x_train, x_test, y_train, y_test):
Trains models, stores model results and images, and saves the models.

Usage:
1. Import the churn_library module.
2. Call the desired functions from the module.

Example:
import churn_library

# Import data
data = churn_library.import_data('data.csv')

# Perform EDA
churn_library.perform_eda(data)

# Perform feature engineering
x_train, x_test, y_train, y_test = churn_library.perform_feature_engineering(data, 'Churn')

# Train models
churn_library.train_models(x_train, x_test, y_train, y_test)

'''

import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, plot_roc_curve
from constants import categorical_columns, numerical_columns

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
logging.basicConfig(
    filename='./logs/results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')




def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    try:
        data_frame = pd.read_csv(pth)
        logging.info("Read file successfully.")
    except FileNotFoundError:
        logging.error("File was not found.")
    return data_frame

def perform_eda(data_frame):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            data_frame: pandas dataframe with new columns for churn
    '''
    try:
        data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        plt.clf()
        data_frame['Churn'].plot(kind='hist').figure.savefig('./images/eda/churn.png')
        plt.clf()
        data_frame['Customer_Age'].plot(kind='hist').figure.savefig(
            './images/eda/customer_Age.png')
        plt.clf()
        data_frame.Marital_Status.value_counts('normalize').plot(
            kind='bar').figure.savefig('./images/eda/Marital_Status.png')
        plt.clf()
        sns.histplot(
            data_frame['Total_Trans_Ct'],
            stat='density',
            kde=True).figure.savefig('./images/eda/total_trans_ct.png')
        plt.clf()
        sns.heatmap(data_frame[numerical_columns].corr(), annot=False, cmap='Dark2_r',
                    linewidths=2).figure.savefig('./images/eda/heat_map.png')
        plt.clf()
        data_frame = encoder_helper(data_frame, categorical_columns, 'Churn')
        logging.info("Successfully performed EDA.")
        return data_frame
    except TypeError as err:
        logging.error("One of the features types can't be plotted.")
        raise err

def encoder_helper(data_frame, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
            [optional argument that could be used for naming
            variables or index y column]

    output:
            data_frame: pandas dataframe with new columns for
    '''
    try:
        for category in category_lst:
            churn_prop = data_frame.groupby(category)[response].mean()
            data_frame[f'{category}_{response}'] = data_frame[category].map(churn_prop)
        logging.info("Successfully encoded columns.")
        return data_frame
    except KeyError as err:
        logging.error("A missing feature has been added to the function.")
        raise err


def perform_feature_engineering(data_frame, response):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name
              [optional argument that could be used
              for naming variables or index y column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    try:
        x_df = pd.DataFrame()
        keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                    'Total_Relationship_Count', 'Months_Inactive_12_mon',
                    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                    'Income_Category_Churn', 'Card_Category_Churn']
        x_df[keep_cols] = data_frame[keep_cols]
        x_train, x_test, y_train, y_test = train_test_split(
            x_df, data_frame[response], test_size=0.3, random_state=42)
        logging.info("Successfully performed feature engineering.")
        return x_train, x_test, y_train, y_test
    except KeyError as err:
        logging.error("A missing feature has been added to the function.")
        raise err


def generate_report_image(report, report_name):
    '''
    A function that takes the text of a classification report and turns it into image
    then saves it in the images directory.
    input:
        report: the classification report of interest
        report_name: the name of the report to be included in the image.
    output:
        None
    '''
    try:
        plt.clf()
        plt.rc('figure', figsize=(5, 5))
        plt.text(
            0.01, 1.25, str(report_name), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.05, str(report), {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig(f'./images/results/{report_name}.png')
        logging.info("Successfully generated report image.")
    except ValueError as err:
        logging.error("The classification report could not be converted to an image.")
        raise err

def create_classification_report_image(y_values,
                                       y_train_preds_lr,
                                       y_train_preds_rf,
                                       y_test_preds_lr,
                                       y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_values: reponse values: a list of (y_train, y_test)
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    try:
        # Random_forests
        generate_report_image(
            classification_report(
                y_values[1],
                y_test_preds_rf),
            'Random Forest Test')
        generate_report_image(
            classification_report(
                y_values[0],
                y_train_preds_rf),
            'Random Forest Train')
        # Logistic Regression
        generate_report_image(
            classification_report(
                y_values[0],
                y_train_preds_lr),
            'Logistic Regression Train')
        generate_report_image(
            classification_report(
                y_values[1],
                y_test_preds_lr),
            'Logistic Regression Test')
        logging.info("Successfully created classification report images.")
    except ValueError as err:
        logging.error("The classification report could not be converted to an image.")
        raise err

def roc_curve_plot(model, model_name, x_test, y_test):
    '''
    creates and stores the roc curve in pth
    input:
            model: model object
            model_name: name of the model
            x_test: X testing data
            y_test: y testing data

    output:
             None
    '''
    try:
        roc_display = plot_roc_curve(model, x_test, y_test)
        roc_display.figure_.savefig(f'./images/results/roc_curve_{model_name}.png')
        logging.info("Successfully created ROC curve.")
    except ValueError as err:
        logging.error("The ROC curve could not be created.")
        raise err


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        names = [x_data.columns[i] for i in indices]
        plt.clf()
        plt.title('Feature Importance')
        plt.ylabel('Importance')
        plt.bar(range(x_data.shape[1]), importances[indices])
        plt.xticks(range(x_data.shape[1]), names, rotation=90)
        plt.savefig(output_pth)
        logging.info("Successfully created feature importance plot.")
    except ValueError as err:
        logging.error("The feature importance plot could not be created.")
        raise err


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    try:
        rfc = RandomForestClassifier(random_state=42, max_features=1.0)
        lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(x_train, y_train)

        lrc.fit(x_train, y_train)

        y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

        y_train_preds_lr = lrc.predict(x_train)
        y_test_preds_lr = lrc.predict(x_test)
        create_classification_report_image(
            [y_train, y_test],
            y_train_preds_lr,
            y_train_preds_rf,
            y_test_preds_lr,
            y_test_preds_rf)
        feature_importance_plot(cv_rfc.best_estimator_, x_train,
                                './images/results/feature_importance.png')
        roc_curve_plot(cv_rfc.best_estimator_, 'random_forests', x_test, y_test)
        roc_curve_plot(lrc, 'logistic_regression', x_test, y_test)

        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc, './models/logistic_model.pkl')
        logging.info("Successfully trained models.")
    except ValueError as err:
        logging.error("The models could not be trained.")
        raise err


if __name__ == "__main__":
    data = import_data('data/bank_data.csv')
    data = perform_eda(data)
    pred_train, pred_test, res_train, res_test = perform_feature_engineering(data, 'Churn')
    train_models(pred_train, pred_test, res_train, res_test)
    logging.info("All functions executed successfully.")
