from sklearn.metrics import classification_report

#accuracy:TP,TN,FP,FN

#Accuracy , Precission , F1 , F2 
def model_evaluate(y_test,y_pred):
    score = classification_report(y_test,y_pred)  #(actual,predicted)

    return score
    