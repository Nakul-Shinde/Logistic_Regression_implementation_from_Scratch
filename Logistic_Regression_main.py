
Log_accuracy_list = []
sklearn_accuracy_list= []


# for loop for iterating 10 times
for i in range(1,11): 
    
    import pandas as pd
    import numpy as np
    from final_code import LogisticRegression
    from final_code import accuracy
    from final_code import z_score
   
 

#-----------------------------------------------------------------------------
#importing data and spliting into training and testing data

#-----------------------------------------------------------------------------
    df = pd.read_csv(r"E:\Machine Learning\Assignment\Assignment_2\Wildfires.txt",sep='\t')

    # takes random data in each iteration
    df = df.sample(frac=1).reset_index(drop=True)

    df.columns =['fire','year','temp','humidity','rainfall','drought_code'
                 ,'buildup_index','day','month','wind_speed']


    dataset_data = df.drop(df.columns[[0]], axis=1)
    
    # Perform feature scaling on dataset
    dataset_data = z_score(dataset_data)
    dataset_data=np.array(dataset_data, dtype=np.float32)

    # performing cleaing operation on labels and converting yes and no
    # into 0's and 1's

    dataset_labels = df['fire']
    dataset_labels = df['fire'].str.strip()
    dataset_labels_dummies = pd.get_dummies(dataset_labels)  
    dataset_labels = dataset_labels_dummies.iloc[:, 1]
    
    # taking the training data size of 2/3 from total dataset
   # X_train, X_test, y_train,y_test = train_test_split(dataset_data,dataset_labels,test_size = 0.334)
    train_size = int(0.6667 * len(df))
    
    #Spliting into training and test data
    X_train = dataset_data[:train_size]
    X_test = dataset_data[train_size:]

    y_train = dataset_labels[:train_size]
    y_test = dataset_labels[train_size:]
    
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

#------------------------------------------------------------------------------
# Model fit and prediction of Logistic Regression

#------------------------------------------------------------------------------
   
    regressor = LogisticRegression( learning_rate=0.001,no_iterations=4500)
    costs = regressor.fit_function( X_train, y_train)
    
    #probabilities is used to store the sigmoid function values which are used for plotting ROC curve
    test_prediction = regressor.predict_function(X_test) 
    
    # calculating accuracy
    Log_accuracy=accuracy(y_test,test_prediction)
    #print('Test Accuracy from our code',Log_accuracy)

    #Log_accuracy_list is used to store the accuarcies for each iteration
    Log_accuracy_list.append(Log_accuracy)
    
#------------------------------------------------------------------------------
# Accuracy using SKLearn code for Logistic Regression

#------------------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression

    logistic =LogisticRegression()
    logistic.fit(X_train,y_train)
    predictions_train = logistic.predict(X_test)
    sklearn_accuracy=accuracy_score(y_test,predictions_train)
    #print('Test Accuracy from sklearn code',sklearn_accuracy)
    
    #sklearn_accuracy_list is used to store the accuarcies for each iteration
    sklearn_accuracy_list.append(sklearn_accuracy)
    

#------------------------------------------------------------------------------
#code to load predicted results and actual test results in excel file

#------------------------------------------------------------------------------
    
    #from pandas import Series, ExcelWriter
    df_actual_predicted_op = pd.DataFrame({ 'Acutal_Test_Results':y_test,'Predicted_Test_Results':test_prediction,'SKLearn_Predicted_Test_Results':predictions_train})

    # create excel writer object
     # create excel writer object
    if(i==1):
        writer = pd.ExcelWriter('E:\Machine Learning\Assignment\Assignment_2\Result_output.xlsx',engine='openpyxl',mode='w')
        # k=i
        #for j in range(k,i+1):
        df_actual_predicted_op.to_excel(writer, sheet_name='sheet '  + str(i))
        writer.save()
    else: 
        writer = pd.ExcelWriter('E:\Machine Learning\Assignment\Assignment_2\Result_output.xlsx',engine='openpyxl',mode='a')
        # k=i
        #for j in range(k,i+1):
        df_actual_predicted_op.to_excel(writer, sheet_name='sheet '  + str(i))
        writer.save()
   
    # save the excel

print('Test results and predicted results are  written successfully to Excel File.')
#writer.close()

#------------------------------------------------------------------------------
#code to load predicted results and actual test results in excel file

#------------------------------------------------------------------------------

df_Accuracy = pd.DataFrame({'Accuracy_Score_Logistic':Log_accuracy_list,'Accuracy_Score_SKLearn':sklearn_accuracy_list})

# create excel writer object
Accuracy_writer = pd.ExcelWriter('E:\Machine Learning\Assignment\Assignment_2\Accuracy_output.xlsx')
df_Accuracy.to_excel(Accuracy_writer)

# save the excel
Accuracy_writer.save()
#Accuracy_writer.close()
print('Accuracy score is written successfully to Excel File.')


#------------------------------------------------------------------------------
#print average of accuracy

#------------------------------------------------------------------------------
from statistics import mean
print(df_Accuracy,"\n")

print('The mean accuracy of our classifier after 10 iteration',mean(Log_accuracy_list))
print('\nThe mean accuracy of SKLearn classifier after 10 iteration',mean(sklearn_accuracy_list))


#------------------------------------------------------------------------------
#ROC curve of our classifier and sklearn classifier 

#------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from sklearn import metrics

x_test_probability=regressor.ROC_Curve_probability(X_test)
x_test_sklearn_probability = logistic.predict_proba(X_test)[::,1]


fpr_local,tpr_local,threshold  = metrics.roc_curve(y_test, x_test_probability.T)
auc_local = metrics.roc_auc_score(y_test, x_test_probability.T)
fpr_sklearn,tpr_sklearn,threshold  = metrics.roc_curve(y_test, x_test_sklearn_probability)
auc_sklearn = metrics.roc_auc_score(y_test, x_test_sklearn_probability)

plt.subplots(1, figsize=(10,6))
plt.plot(fpr_local, tpr_local,label ="ROC curve of local model,auc="+str(auc_local))
plt.legend(loc="lower right")
plt.plot(fpr_sklearn,tpr_sklearn,label="ROC curve of Sklearn Model,auc="+str(auc_sklearn))
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.legend()
plt.show()



#-----------------------------------------------------------------------------
#Plot for cost vs numver of iterations

#-----------------------------------------------------------------------------
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title('Cost reduction over time')
plt.show()
