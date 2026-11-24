
from src.data_ingestion import data_ingestion
from src.data_preprocessing import data_preprocessing
from src.model_building import model_build
from src.model_prediction import model_predict
from src.model_evaluation import model_evaluate

#step 1: Data ingestion 
def main():
    df= data_ingestion()
    print(df.columns)

#step 2 : Data Preprocessing

    X_train_resample, X_test_transform, y_train_resample, y_test= data_preprocessing(df)
    print(X_train_resample)    

#Step 3 : Model Building

    model=model_build(X_train_resample, X_test_transform,y_train_resample)

#Step 4 : Model Prediction

    y_pred = model_predict(model,X_test_transform)

#Step 5 : Model Evaluate

    score = model_evaluate(y_test,y_pred)

    print(f'Model Classification Report : {score}')

main()