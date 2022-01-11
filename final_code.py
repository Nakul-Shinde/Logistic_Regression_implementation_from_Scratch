

import pandas as pd
import numpy as np

#------------------------------------------------------------------------------
#Logistic regression classifier that contents
#1. Sigmoid function
#2. gradident descent function
#3. Fit function
#4. predict function
#------------------------------------------------------------------------------
class LogisticRegression:

    def __init__(self, learning_rate, no_iterations):
        self.learning_rate = learning_rate
        self.no_iterations = no_iterations
       

    # Calulate the probability values
    def sigmoid_function(self,formula):
       result = 1/(1+np.exp(-formula))
       return result
    

    # intially the probability value is calculated for each feature which is then 
    # used to calculate the cost.
    # This cost is then minimumized by using gradient function
    def gradient_cost_function(self,weights, bias, x_value, y_value):
        m = x_value.shape[0]
        
        #Prediction
        final_result = self.sigmoid_function(np.dot(weights,x_value.T)+bias)
        #cost
        y_value_Transpose = y_value.T
        cost = (-1/m)*(np.sum((y_value_Transpose*np.log(final_result)) + ((1-y_value_Transpose)*(np.log(1-final_result)))))
        #
     
        #Gradient calculation
        d_weight = (1/m)*(np.dot(x_value.T, (final_result-y_value.T).T))
        d_bias = (1/m)*(np.sum(final_result-y_value.T))
        
        gradients = {"d_weight": d_weight, "d_bias": d_bias}
        
        return gradients, cost
    

    # intially the value of weight and bias is set to zero.
    # for every iteration we first calculate the weight and bias by using graident 
    #descent cost function these values are stored in dw and db. After that the
    #weight and bias is updated.
    def fit_function(self, X_train, Y_train):
        
        #Get number of features
        no_of_features = X_train.shape[1]
        weights = np.zeros((1,no_of_features))
        bias = 0
        costs = []
        
        for i in range(self.no_iterations):
            #
            gradients, cost = self.gradient_cost_function(weights,bias,X_train,Y_train)
            #
            d_weight = gradients["d_weight"]
            d_bias = gradients["d_bias"]
             #weight update
            weights = weights - (self.learning_rate * (d_weight.T))
            bias = bias - (self.learning_rate * d_bias)
                     
            if (i % 100 == 0):
                    costs.append(cost)
                   
        
        #final parameters
        coefficient = {"Weights": weights, "Bias": bias}
        self.weights = coefficient["Weights"]
        self.bias = coefficient["Bias"]
        return  costs
        

    #initially the diamension of test matrix has been extracted and assigned to m.
    #Using standard equ. of line Z= βo+ β1x, the value of z is calulated
    #By calling sigmoid function the values of equation got converted in probabilities.
    #value is then compared if it is greater than 0.5 it returns 1 else 0.
   
    
    def predict_function(self,pred_value):
        m= pred_value.shape[0]
        Y=np.dot(self.weights,pred_value.T)+self.bias
        sigmoid_prediction = self.sigmoid_function(Y)
        
        y_pred = [None] * m          
        for i in range(sigmoid_prediction.shape[1]):
            if sigmoid_prediction[0][i] > 0.5:
                y_pred[i] = 1
            else: 
                y_pred[i] = 0
                
        return y_pred
    

    #By using the of line Z= βo+ β1x, the value of z is calulated
    #This value is then returned 
    
    def ROC_Curve_probability(self,X_test):
        Y=np.dot(self.weights,X_test.T)+self.bias
        Probability_value = self.sigmoid_function(Y)
        
     
                
        return Probability_value


#------------------------------------------------------------------------------
#Zscore function for performing Feature scaling.

#------------------------------------------------------------------------------
def z_score(df):
    
    for column in df.columns:
        df[column] = (df[column] - df[column].mean()) / df[column].std()

    return df


#------------------------------------------------------------------------------
#Function to calculate Accuracy

#------------------------------------------------------------------------------

def accuracy(labels, predictions):


  assert len(labels) == len(predictions)
  
  correct = 0
  for label, prediction in zip(labels, predictions):
    if label == prediction:
      correct += 1 
  
  score = correct / len(labels)
  return score
