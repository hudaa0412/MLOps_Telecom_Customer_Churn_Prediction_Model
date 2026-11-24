from sklearn.ensemble import RandomForestClassifier

def model_build(X_train_resample, X_test_transform,y_train_resample):
    model=RandomForestClassifier().fit(X_train_resample,
                                       y_train_resample) #Seen Data

    return model