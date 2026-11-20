import pandas as pd

def data_ingestion():

    '''
    * Title: Telecom Customer Churn Predection Model
    
    *Independent column[X]:['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges', ],


    *dependent columns[y]:'Churn'

    * Problem Statement:
    1. To predict the churn rate
    2. Use Clustering approach to predict and clssify the cluster pattern such as {l1,l2 & l3}


    '''

    df = pd.read_csv(r'C:\MLOps_Telecom_Customer_Churn_Prediction_Model\data\raw\churn.csv')

    return df