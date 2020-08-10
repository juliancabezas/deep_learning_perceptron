import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

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

def perceptron_train(x,y,n_iter):

    w = []
    for i in range(x.shape[1]):
        w.append(0.0)

    predicted = perceptron_prediction_total(x,w)

    #step = 1 / len(x)
    step = 0.1
    #step = 1.0

    for iter in range(n_iter):

        error = 0
        predic = perceptron_prediction_total(x,w)
        acc = perceptron_accuracy(y,perceptron_prediction_total(x,w))
        #print(perceptron_accuracy(y,perceptron_prediction_total(x,w)))

        for i in range(len(x)):

            prediction = perceptron_prediction(x[i][:],w)

            #error = y[i] - prediction 

            for j in range(len(w)):

                #prediction = perceptron_prediction(x[i][:],w) 			

                #w[j] = w[j]+(step*error*x[i][j]) 

                if (prediction == y[i]):
                    error = 0
                else:
                    error = 1

                w[j] = w[j]+(step*y[i]*x[i][j]*error)  
                #w[j] = w[j]+(step*x[i][j]*error)  
                #print(w[j])

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

    #diabetes.insert(0, 'Intercept', 1.0)

    print(diabetes)

    # Generate lists of lists
    x = diabetes.iloc[:, :-1].values.tolist()
    y = diabetes.iloc[:, -1].values.tolist()

    x = diabetes.iloc[:, :-1].values
    y = diabetes.iloc[:, -1].values

    print(np.mean(y))

    # Generate test and train datasets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23,stratify=y,shuffle=True)

    #print(y_test)

    #print(np.mean(y_test))
    #print(len(y_test))

    k_fold_strat = StratifiedKFold(n_splits=4, random_state=23,shuffle=True)
    #k_fold_strat.get_n_splits(x, y)



    # Iterate thorgh the folds
    for kfold_train_index, kfold_test_index in k_fold_strat.split(x, y):
        kfold_x_train, kfold_x_test = x[kfold_train_index][:], x[kfold_test_index][:]
        kfold_y_train, kfold_y_test = y[kfold_train_index], y[kfold_test_index]

        #initial_weights = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
        trained_weights = perceptron_train(kfold_x_train,kfold_y_train,300)
        predicted = perceptron_prediction_total(kfold_x_test,trained_weights)
        print(trained_weights)
        print(perceptron_accuracy(y,kfold_y_test))

    




    # Training and validation using k-fold






    initial_weights = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]


    #


    #weights = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    #weights = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
    #weights = [-1.00, -1.00, -1.00, -1.00, -1.00, -1.00, -1.00, -1.00]


    #final_weights = perceptron_train(x,y,weights,100)
    #predicted = perceptron_prediction_total(diabetes.iloc[:, :-1].values.tolist(),final_weights)
    #print(final_weights)
    #print(perceptron_accuracy(y,predicted))

    # Initial weights




if __name__ == '__main__':
	main()





