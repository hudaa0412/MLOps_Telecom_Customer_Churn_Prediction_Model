from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE , ADASYN 

def data_preprocessing(df):
    # Step1: Drop duplicated values
    # Step2: Feature Engineering (Optional)**
    # Step3: Split the dataset into Indepedent columns {X} and target column {y}
    # Step4: Split the dataset into {train} Seen and {test} Unseen data
    # Step5: Build EnE Pipeline for numerical and categorical 
    # Step6: Build numerical pipeline with imputation and scaling
    # Step7: Build categorical pipeline with imputation and encoding
    # Step8: Concat numerical and categorical pipeline using ColumnTransformer
    # Step9: fit transformer on train and test
    # Step10: fit_resamlple the data so that target column balance
    # Step11: Return the preprocess train and test : X_train,X_test,y_train,y_test
    #-------------------------------------------------------------------------------------------
    # STEP 1 : Ensuring drop the duplicate values.
    df.drop_duplicates(inplace = True)

    # Step 2 : Feature Engineering (Optional).

    # Step 3 : Split the dataset into  X and y.

    X = df.drop(columns=['customerID','Churn'])    # Independent columns
    y = df['Churn']                   # Dependent column

    # Step 4 : Split the dataset into train(Seen) and test(Test) dataset.

    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size = 0.3,
                                                        random_state = 1)

    # Step 5 : Segregate the pipeline into numerical and categorical.

    # Segregate the numerical and categorical columns.
    numerical_col = X_train.select_dtypes(exclude = 'object').columns
    categorical_col = X_train.select_dtypes(include = 'object').columns

    # Step 6 : Use numerical_pipeline

    numerical_pipeline = Pipeline([('Imputation',SimpleImputer(strategy = 'median')),
                                   ('Scaling',MinMaxScaler())])

    # Step 7 : Use categorical_pipeline
    
    categorical_pipeline = Pipeline([('Imputation',SimpleImputer(strategy = 'most_frequent')),
                                     ('Encoding',OneHotEncoder(handle_unknown = 'ignore'))])

    # Step 8 : Use columntransformer to concat numerical_pipeline and categorical_pipeline
     
    transformer = ColumnTransformer([('numerical_pipeline',
                                        numerical_pipeline,
                                        numerical_col),
                                        ('categorical_pipeline',
                                        categorical_pipeline,
                                        categorical_col)])

    # Step 9 : Fit the transformer on seen and unseen dataset.

    X_train = transformer.fit_transform(X_train)   #Seen dataset
    X_test = transformer.transform(X_test)         #Unseen dataset

    #Step 10 : Use target column to balance technique
    '''
    * Generally used with clssification model
    * Target column balance technique basically classified into two...
       - over_sampling technique
       - under_sampling technique
    '''
    smote = SMOTE(random_state=1)
    X_train,y_train = SMOTE().fit_resample(X_train,y_train)

    # Step 11 : Return the preprocess data value

    return X_train, X_test, y_train, y_test