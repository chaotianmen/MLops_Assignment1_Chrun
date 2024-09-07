"""
This module contains functions for dealing with Churn data, which includes data import, EDA, encoding, 
feature engineering, and model training.
"""


# import libraries
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    Perform EDA on df and save figures to images folder

    input:
            df: pandas dataframe

    output:
            None
    '''

    # Define categorical and quantitative columns
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    quant_columns = [
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
        'Avg_Utilization_Ratio'
    ]

    # Create images directory if it doesn't exist
    if not os.path.exists('images'):
        os.makedirs('images')

    # Basic DataFrame information
    print(f"DataFrame shape: {df.shape}")
    print(f"Missing values in each column:\n{df.isnull().sum()}")
    print(f"DataFrame description:\n{df.describe()}")

    # Add churn column based on 'Attrition_Flag'
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Plot churn distribution
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.title('Churn Distribution')
    plt.savefig('./images/eda/churn_distribution.png')
    plt.close()

    # Plot customer age distribution
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.title('Customer Age Distribution')
    plt.savefig('./images/eda/customer_age_distribution.png')
    plt.close()

    # Plot marital status distribution
    plt.figure(figsize=(20, 10))
    df['Marital_Status'].value_counts('normalize').plot(kind='bar')
    plt.title('Marital Status Distribution')
    plt.savefig('./images/eda/marital_status_distribution.png')
    plt.close()

    # Plot distribution of Total Transactions Count with KDE
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.title('Distribution of Total Transaction Count')
    plt.savefig('./images/eda/total_transaction_count_distribution.png')
    plt.close()

    # Plot heatmap of correlations between quantitative variables
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        df[quant_columns].corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.title('Heatmap of Quantitative Features Correlation')
    plt.savefig('./images/eda/quantitative_features_correlation_heatmap.png')
    plt.close()


def encoder_helper(df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    df0 = df.copy()
    df0[response] = df0['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    for column in category_lst:
        churn_mean = df0.groupby(column)[response].mean()
        df0[f'{column}_{response}'] = df0[column].map(churn_mean)
    return df0


def perform_feature_engineering(
        df,
        keep_cols,
        test_size=0.3,
        random_state=42,
        response='Churn'):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    X = df[keep_cols]
    y = df[response]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    Produces classification report for training and testing results and stores report as image
    in images/results folder.

    Input:
        y_train: training response values
        y_test: test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest

    Output:
        None
    '''

    # Create the output directory if it doesn't exist
    output_path = './images/results'
    os.makedirs(output_path, exist_ok=True)

    # Random Forest classification report
    rf_test_report = classification_report(
        y_test, y_test_preds_rf, output_dict=False)
    rf_train_report = classification_report(
        y_train, y_train_preds_rf, output_dict=False)

    # Logistic Regression classification report
    lr_test_report = classification_report(
        y_test, y_test_preds_lr, output_dict=False)
    lr_train_report = classification_report(
        y_train, y_train_preds_lr, output_dict=False)

    # Save reports as images
    for model_name, train_report, test_report in zip(
        ['Random Forest', 'Logistic Regression'],
        [rf_train_report, lr_train_report],
        [rf_test_report, lr_test_report]
    ):
        plt.figure(figsize=(10, 6))
        plt.text(0.01, 0.05, f"Train Report:\n\n{train_report}",
                 {'fontsize': 12}, fontproperties='monospace')
        plt.text(0.01, 0.6, f"Test Report:\n\n{test_report}",
                 {'fontsize': 12}, fontproperties='monospace')
        plt.axis('off')

        # Save the image
        image_path = os.path.join(output_path,
                                  f"{model_name}_classification_report.png")
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()


def roc_curves_image(y_test, lr_model, rf_model, X_test):
    '''
    Plots and saves ROC curves for both Logistic Regression and Random Forest models.

    Input:
        y_test: True test response values
        lr_model: Trained Logistic Regression model
        rf_model: Trained Random Forest model
        X_test: Test feature dataset

    Output:
        None
    '''
    plt.figure(figsize=(15, 8))
    ax = plt.gca()

    # Logistic Regression ROC curve
    RocCurveDisplay.from_estimator(
        lr_model,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8,
        name='Logistic Regression')

    # Random Forest ROC curve
    RocCurveDisplay.from_estimator(
        rf_model,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8,
        name='Random Forest')

    # Save the ROC curves plot
    plt.title("ROC Curves: Logistic Regression vs Random Forest")
    plt.savefig('./images/results/roc_curves.png', bbox_inches='tight')
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    Creates and stores the feature importances plot using both SHAP and model's feature_importances_.

    Input:
        model: model object containing feature_importances_ (RandomForest, XGBoost, etc.)
        X_data: pandas dataframe of X values (input features)
        output_pth: path to store the figures

    Output:
        None
    '''

    # Ensure the output directory exists
    os.makedirs(output_pth, exist_ok=True)

    # SHAP plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)

    # Save SHAP summary plot as a bar chart
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    shap_output_path = os.path.join(output_pth, "shap_summary_plot.png")
    plt.savefig(shap_output_path, bbox_inches='tight')
    plt.close()

    # Feature importance from model's feature_importances_
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = [X_data.columns[i] for i in indices]

    # Plot feature importance
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), feature_names, rotation=90)
    plt.ylabel('Importance')

    # Save the feature importance plot
    importance_output_path = os.path.join(
        output_pth, "feature_importance_plot.png")
    plt.savefig(importance_output_path, bbox_inches='tight')
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=5000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt', None],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    roc_curves_image(y_test, lr_model, rfc_model, X_test)

    X_data = pd.concat([X_train, X_test], axis=0, ignore_index=True)

    feature_importance_plot(rfc_model, X_data, './images/results')

    pass
