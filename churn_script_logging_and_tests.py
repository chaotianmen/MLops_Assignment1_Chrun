"""
This module contains test functions for the churn library, which includes data import, EDA, encoding, 
feature engineering, and model training.
"""

import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data, pth):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data(pth)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        return

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("Testing import_data: The DataFrame has rows and columns")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")


def test_eda(perform_eda, import_data, pth):
    '''
    test perform eda function
    '''
    df = import_data(pth)
    logging.info("Testing import_data: SUCCESS")

    try:
        perform_eda(df)
        logging.info("Finishing eda work")
        # Check if output images are generated
        assert os.path.exists('./images/eda/churn_distribution.png')
        assert os.path.exists('./images/eda/customer_age_distribution.png')
        assert os.path.exists('./images/eda/marital_status_distribution.png')
        assert os.path.exists(
            './images/eda/total_transaction_count_distribution.png')
        assert os.path.exists(
            './images/eda/quantitative_features_correlation_heatmap.png')
        logging.info("all eda visualization have been saved")
    except AssertionError as err:
        logging.error("at least one image is not saved")
        raise err


def test_encoder_helper(
        encoder_helper,
        import_data,
        pth,
        category_lst,
        response):
    '''
    Test encoder_helper function
    '''
    # Step 1: Import the data
    try:
        df = import_data(pth)
        logging.info("Testing import_data: SUCCESS")
    except Exception as e:
        logging.error("Testing import_data: FAILED with error: %s", e)
        return

    # Step 2: Test encoder_helper function
    try:
        # Call encoder_helper
        df_new = encoder_helper(df, category_lst, response)
        logging.info("Testing encoder_helper: SUCCESS")

        # Test the output shape
        new_column_number = df.shape[1] + 1 + len(category_lst)
        assert df_new.shape[1] == new_column_number, \
            "Expected %d columns, but got %d" % (new_column_number, df_new.shape[1])
        logging.info(
            "encoder_helper Result Checking: SUCCESS, correct number of columns created")

    except KeyError as ke:
        logging.error("wrong column name: %s", ke)

    except AssertionError as ae:
        logging.error(
            "encoder_helper Result Checking: FAILED, column numbers mismatch: %s", ae)

    except Exception as e:
        logging.error("Unexpected error in encoder_helper: %s", e)


def test_perform_feature_engineering(
        perform_feature_engineering,
        import_data,
        encoder_helper,
        pth,
        category_lst,
        response,
        keep_cols,
        test_size,
        random_state):
    """
    Test perform_feature_engineering
    """
    # Step 1: Import and process the data
    try:
        df = import_data(pth)
        logging.info("Testing import_data: SUCCESS")
        df_new = encoder_helper(df, category_lst, response)
        logging.info("Testing encoder_helper: SUCCESS")
    except Exception as e:
        logging.error("Import and encode data: FAILED with error: %s", e)
        return

    # Step 2: Test feature engineering function
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            df=df_new,
            keep_cols=keep_cols,
            test_size=test_size,
            random_state=random_state,
            response=response
        )
        logging.info("Testing perform_feature_engineering: SUCCESS")

        # Check dimensions
        logging.info("X_train shape: %s", X_train.shape)
        logging.info("X_test shape: %s", X_test.shape)
        logging.info("y_train shape: %s", y_train.shape)
        logging.info("y_test shape: %s", y_test.shape)

    except KeyError as ke:
        logging.error("Key error in feature engineering: %s", ke)
    except Exception as e:
        logging.error("Feature engineering: FAILED with error: %s", e)


def test_train_models(
        train_models,
        import_data,
        encoder_helper,
        perform_feature_engineering,
        pth,
        category_lst,
        response,
        keep_cols,
        test_size,
        random_state):
    '''
    Test the train_models function to ensure the models are trained, saved, and output correctly.

    Input:
        train_models: function to train models.
        import_data: function to import the data.
        encoder_helper: function to encode categorical columns.
        perform_feature_engineering: function to split data into train/test sets.
        pth: path to the data file.
        category_lst: list of categorical features to encode.
        response: target column name.
        keep_cols: features to keep.
        test_size: test split size.
        random_state: random seed for reproducibility.

    Output:
        None
    '''
    # Step 1: Import and process the data
    try:
        df = import_data(pth)
        logging.info("Testing import_data: SUCCESS")
        df_new = encoder_helper(df, category_lst, response)
        logging.info("Testing encoder_helper: SUCCESS")
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            df=df_new,
            keep_cols=keep_cols,
            test_size=test_size,
            random_state=random_state,
            response=response
        )
        logging.info("Testing perform_feature_engineering: SUCCESS")

    except Exception as e:
        logging.error("Import and process data: FAILED with error: %s", e)
        return

    # Step 2: Train the models and check outputs
    try:
        train_models(X_train, X_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")
    except Exception as e:
        logging.error("Testing train_models: FAILED with error: %s", e)
        return

    # Step 3: Validate model files
    try:
        # Check if models are saved
        assert os.path.exists(
            './models/rfc_model.pkl'), "Random Forest model not saved"
        assert os.path.exists(
            './models/logistic_model.pkl'), "Logistic Regression model not saved"

        logging.info("Model files check: SUCCESS")
    except Exception as e:
        logging.error("Model files check: FAILED with error: %s", e)
        return

    # Step 4: Validate image outputs
    try:
        # Check if the images were saved
        assert os.path.exists(
            './images/results/feature_importance_plot.png'), "Feature importance plot not saved"
        assert os.path.exists(
            './images/results/roc_curves.png'), "ROC curve image not saved"
        assert os.path.exists(
            './images/results/shap_summary_plot.png'), "SHAP summary plot not saved"

        logging.info("Image files check: SUCCESS")
    except Exception as e:
        logging.error("Image files check: FAILED with error: %s", e)


if __name__ == "__main__":
    test_import(cls.import_data, "./data/bank_data.csv")
    test_import(cls.import_data, "./data/bnk_data0.csv")
    test_eda(cls.perform_eda, cls.import_data, "./data/bank_data.csv")
    test_encoder_helper(
        cls.encoder_helper,
        cls.import_data,
        pth="./data/bank_data.csv",
        category_lst=[
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'],
        response='Churn')
    test_encoder_helper(
        cls.encoder_helper,
        cls.import_data,
        pth="./data/bank_data.csv",
        category_lst=[
            'Wrong_name',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'],
        response='Churn')

    the_columns = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    test_perform_feature_engineering(
        cls.perform_feature_engineering,
        cls.import_data,
        cls.encoder_helper,
        pth="./data/bank_data.csv",
        category_lst=[
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'],
        response='Churn',
        keep_cols=the_columns,
        test_size=0.3,
        random_state=42)

    the_columns = ['Not exist', 'Dependent_count', 'Months_on_book',
                   'Total_Relationship_Count']
    test_perform_feature_engineering(
        cls.perform_feature_engineering,
        cls.import_data,
        cls.encoder_helper,
        pth="./data/bank_data.csv",
        category_lst=[
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'],
        response='Churn',
        keep_cols=the_columns,
        test_size=0.3,
        random_state=42)

    the_columns = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    test_train_models(
        cls.train_models,
        cls.import_data,
        cls.encoder_helper,
        cls.perform_feature_engineering,
        pth="./data/bank_data.csv",
        category_lst=[
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'],
        response='Churn',
        keep_cols=the_columns,
        test_size=0.3,
        random_state=42)
