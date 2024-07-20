import os
import logging
import churn_library_solution as cls
from pathlib import Path

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
        return df
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda, data):
	'''
	test perform eda function
	'''
    try:
        df = perform_eda(data)
        logging.info("Testing perform_eda: SUCCESS")
        return df
    except TypeError as err:
        logging.error("Testing perform_eda: One of the features types can't be plotted.")
        raise err
    try:
        assert len(list(Path('./images/eda').glob('*.png'))) == 5
        logging.info("Testing perform_eda: Graphs have been saved successfully.")
    except AssertionError as err:
        logging.error("The graphs have not been saved successfully.")
        raise err


def test_encoder_helper(encoder_helper, data_frame, categorical_columns, result):
	'''
	test encoder helper
	'''
    try:
        df = encoder_helper(data_frame, categorical_columns, result)
        logging.info("Testing encoder_helper: SUCCESS.")
        return df
    except TypeError as err:
        logging.error("Testing encoder_helper: A feature has an incompatible data type.")
        raise err
        
    try:
        assert len(list(df.columns)) > len(list(data_frame.columns))
        logging.info("Testing encoder_helper: Columns have been added successfully.")
    except AssertionError as err:
        logging.error("Testing encoder_helper: the encoded columns have not been added.")
        raise err


def test_perform_feature_engineering(perform_feature_engineering, data, result):
	'''
	test perform_feature_engineering
	'''
    try:
        pred_train, pred_test, res_train, res_test = perform_feature_engineering(data, 'Churn')
        logging.info("Testing perform_feature_engineering: SUCCESS")
        return pred_train, pred_test, res_train, res_test
    except KeyError as err:
        logging.error("Testing perform_feature_engineering: A missing feature has been added to the function.")
        raise err


def test_train_models(train_models, pred_train, pred_test, res_train, res_test):
	'''
	test train_models
	'''
    try:
        train_model(pred_train, pred_test, res_train, res_test)
        logging.info("Testing train_models: SUCCESS")
    except InvalidParameterError as err:
        logging.error("Testing train_model: An error with the model parameters occurred.")
        raise err
    
    try:
        assert len(list(Path('./images/results').glob('*.png'))) > 0
        logging.info("Testing train_model: SUCCESS - All graphs have been saved.")
    except AssertionError as err:
        logging.error("Testing train_model: Graphs have not been saved.")
        raise err


if __name__ == "__main__":
	df = test_import("./data/bank_data.csv")
    df = test_eda(df)
    pred_train, pred_test, res_train, res_test = test_perform_feature_engineering(data, 'Churn')
    test_train_models(pred_train, pred_test, res_train, res_test)