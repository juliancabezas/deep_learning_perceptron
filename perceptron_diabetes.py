import numpy as np
import pandas as pd

# Predict the output for a single sample
def perceptron_prediction(input,w):

    total_sum = 0.0
    tuples_input_w = zip(input,w)
    prediction = 0.0

    for input_row,weight in tuples_input_w:

        # This will multiply the tuples and add them
        partial_sum = input_row * weight

        total_sum = total_sum + partial_sum
    
    if (total_sum >= 0.0):
        prediction = 1.0
    else:
        prediction = -1.0

    return prediction

# Predict the output for the complete dataset
def perceptron_prediction_total(input_matrix,w):

    predicted = []

    for i in range(len(input_matrix)):
        prueba = input_matrix[i][:]
        predicted_value = perceptron_prediction(input_matrix[i][:],w)
        predicted.append(predicted_value)
    
    return predicted



# Get the error of the perceptron
def perceptron_accuracy(truth,predicted):

    correct = 0

    for i in range(len(predicted)):
        if (predicted[i] == truth[i]):
            correct = correct +1

    return correct / len(predicted)

def perceptron_train(x,y,w,n_iter):

    predicted = perceptron_prediction_total(x,w)

    #step = 1 / len(x)
    step = 0.5

    for iter in range(n_iter):

        error = 0

        for i in range(len(x)):

            prediction = perceptron_prediction(x[i][:],w)

            #error = y[i] - prediction 

            for j in range(len(w)): 			

                #w[j] = w[j]+(step*error*x[i][j]) 

                if (prediction == y[i]):
                    correct = 0
                else:
                    correct = 1

                w[j] = w[j]+(step*y[i]*x[i][j]*correct) 

    return w
        

# scale data to have it between a and b, using the equation:
# x(scaled) = (b-a) (x - min(x) / max(x) - min(x)) + a
# https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
def scale_data(x,lower_limit,upper_limit):

    maximum = np.max(x)
    minimum = np.min(x)
    scaled = []

    for i in range(len(x)):
        x_scaled = ((upper_limit - lower_limit)*((x[i] - minimum) / (maximum - minimum))) + lower_limit
        scaled.append(x_scaled)

    return scaled


def main():

    diabetes = pd.read_csv("datasets_228_482_diabetes.csv")

    # Recode the output column to get -1 and 1 output values
    diabetes['Outcome'] = np.where(diabetes['Outcome'] == 0, -1, diabetes['Outcome'])


    # Scale the variables
    diabetes['Pregnancies'] = scale_data(diabetes['Pregnancies'],-1,1)
    diabetes['Glucose'] = scale_data(diabetes['Glucose'],-1,1)
    diabetes['BloodPressure'] = scale_data(diabetes['BloodPressure'],-1,1)
    diabetes['SkinThickness'] = scale_data(diabetes['SkinThickness'],-1,1)
    diabetes['Insulin'] = scale_data(diabetes['Insulin'],-1,1)
    diabetes['BMI'] = scale_data(diabetes['BMI'],-1,1)
    diabetes['DiabetesPedigreeFunction'] = scale_data(diabetes['DiabetesPedigreeFunction'],-1,1)
    diabetes['Age'] = scale_data(diabetes['Age'],-1,1)

    print(diabetes)

    weights = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    #weights = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
    #weights = [-1.00, -1.00, -1.00, -1.00, -1.00, -1.00, -1.00, -1.00]
    #weights = [0.25, 0.10, 0.5, 1.0, 1.0,1.00, 0.1, 1.00]

    #predicted = perceptron_prediction_total(diabetes.iloc[:, :-1].values.tolist(),weights)

    #print(predicted)

    x = diabetes.iloc[:, :-1].values.tolist()
    
    y = diabetes.iloc[:, -1].values.tolist()


    final_weights = perceptron_train(x,y,weights,10000)

    predicted = perceptron_prediction_total(diabetes.iloc[:, :-1].values.tolist(),final_weights)

    print(final_weights)

    print(perceptron_accuracy(y,predicted))

    # Initial weights




if __name__ == '__main__':
	main()





