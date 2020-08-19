###################################
# Julian Cabezas Pena
# Deep Learning Fundamentals
# University of Adelaide
# Assingment 1
# Single Perceptron implementqation and testiong in the PIMA diabetes data
####################################

# Import libraries
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, fbeta_score

#### Predict the output for a single row
# x: Values for parameters of a single row of the input data
# w: weights of each parameter
def perceptron_prediction(x,w):

    # Initialize variables
    total_sum = 0.0
    prediction = 0.0

    # Calculate the dow product between x and w (weighted sum)
    total_sum = np.dot(x,w)
    
    # Sign function, if the weighted sum is greater than 0 the output is 1, else -1
    if (total_sum >= 0.0):
        prediction = 1.0
    else:
        prediction = -1.0

    return prediction

#### Predict the output for the complete dataset
# x_matrix: Values of the predictors in matrix form
# w: weights of each parameter
def perceptron_prediction_total(x_matrix,w):

    # The function returns a list of predicted values
    predicted = []

    # Loop through the matrix
    for i in range(len(x_matrix)):

        predicted_value = perceptron_prediction(x_matrix[i][:],w)
        predicted.append(predicted_value)
    
    return predicted

# Get the error of the perceptron
def perceptron_accuracy(truth,predicted):

    correct = 0

    for i in range(len(predicted)):
        if (predicted[i] == truth[i]):
            correct = correct +1

    return correct / len(predicted)

# Train the perceptron weights
def perceptron_train(x,y,n_iter,step):

    # generate the initial weights
    #w = []
    #for i in range(x.shape[1]):
    #    w.append(0.0)

    # Set random seed
    np.random.seed(50)

    # generate the initial weights for the input to hidden layer
    w = np.random.normal(0.0, 0.1, x.shape[1])

    # Make n_iter iterations to trin the weights (also called epochs)
    for iter in range(n_iter):

        error = 0

        # Go thorugh the rows of the matrix
        for i in range(len(x)):

            # Predict the result for the single row
            prediction = perceptron_prediction(x[i][:],w)

            # Loop trough the weights
            for j in range(len(w)):

                # If the prediction is not equial to the true value, the error will be set to 1 to make a gradient
                if (prediction == y[i]):
                    error = 0
                else:
                    error = 1

                # Gradient descent on the weitghts
                w[j] = w[j]+(step*y[i]*x[i][j]*error)  

    return w
        

# scale data to have it between a and b, using the equation:
# x(scaled) = (b-a) (x - min(x) / max(x) - min(x)) + a
# https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
def scale_data(x,lower_limit,upper_limit):

    maximum = np.max(x)
    minimum = np.min(x)
    scaled = []

    # Scale the data to fit between the lower and upper limit that was stablished
    for i in range(len(x)):
        x_scaled = ((upper_limit - lower_limit)*((x[i] - minimum) / (maximum - minimum))) + lower_limit
        scaled.append(x_scaled)

    return scaled


# Main function, it will preprocess the data, create train and test data,
# tune the parameters of the perceptron and evaluate the final model
def main():

    # Read the database using pandas
    diabetes = pd.read_csv("Input_Data/datasets_228_482_diabetes.csv")

    # Recode the output column to get -1 and 1 output values
    diabetes['Outcome'] = np.where(diabetes['Outcome'] == 0, -1, diabetes['Outcome'])

    # Preprocessing
    # Replace the missing blood pressure values by the mean
    bp = diabetes['BloodPressure'][diabetes['BloodPressure']!=0]
    mean_bp = bp.mean()
    diabetes['BloodPressure'] = np.where(diabetes['BloodPressure'] == 0, mean_bp, diabetes['BloodPressure'])

    # Replace the BMI missing values by their mean
    bmi = diabetes['BMI'][diabetes['BMI']!=0]
    mean_bmi = bmi.mean()
    diabetes['BMI'] = np.where(diabetes['BMI'] == 0, mean_bmi, diabetes['BMI'])

    # Scale the variables between -1 and +1
    diabetes['Pregnancies'] = scale_data(diabetes['Pregnancies'],-1,1)
    diabetes['Glucose'] = scale_data(diabetes['Glucose'],-1,1)
    diabetes['BloodPressure'] = scale_data(diabetes['BloodPressure'],-1,1)
    diabetes['SkinThickness'] = scale_data(diabetes['SkinThickness'],-1,1)
    diabetes['Insulin'] = scale_data(diabetes['Insulin'],-1,1)
    diabetes['BMI'] = scale_data(diabetes['BMI'],-1,1)
    diabetes['DiabetesPedigreeFunction'] = scale_data(diabetes['DiabetesPedigreeFunction'],-1,1)
    diabetes['Age'] = scale_data(diabetes['Age'],-1,1)
    diabetes.insert(2, 'Intercept', 1.0)


    # Print a subset of the database
    print(diabetes)

    # Generate lists of lists
    x = diabetes.iloc[:, :-1].values
    y = diabetes.iloc[:, -1].values

    # Generate test and train datasets, 20% goes into testing
    # Stratified splitting is used to ensure a fixed proportion of diabetics/non-diabetics in the train and test datasets (around 30%)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23,stratify=y,shuffle=True)


    # Tuning of the n_iter (epochs) and step parameters using 4-fold cross validation on the trainining database
    # If the parameters were already tuned (the file exists)
    if not os.path.exists('Results/grid_search_perceptron.csv'):

        print("Parameter tuning not detected, tuning step and number of epochs parameters")

        # set up a tratified 4-fold partition of the train data
        k_fold_strat = StratifiedKFold(n_splits=4, random_state=23,shuffle=True)

        # Test steps for the gradient descent from 0.1 to 1
        step_array = np.arange(start=0.1, stop=1.1, step=0.1)

        # Test different numbers of iterations from 10 110
        iter_array = np.arange(start=10, stop=110, step=10)

        # Store the partial results in lists
        step_full = []
        n_iter_full = []
        acc_full = []
        kappa_full = []
        f1_full = []

        # Loop trough the different combinations of step and number of iterations
        for step in step_array:

            for n_iter in iter_array:
                
                #Store partial results for accuracy, cohen kappa and F1
                acc = []
                kappa = []
                f1 = []

                # Iterate thorgh the 4 folds
                for kfold_train_index, kfold_test_index in k_fold_strat.split(x_train, y_train):
                    
                    # Get the split into train and test
                    kfold_x_train, kfold_x_test = x[kfold_train_index][:], x[kfold_test_index][:]
                    kfold_y_train, kfold_y_test = y[kfold_train_index], y[kfold_test_index]

                    # Train the perceptron and get the predicted values
                    trained_weights = perceptron_train(kfold_x_train,kfold_y_train,n_iter=n_iter,step=step)
                    predicted = perceptron_prediction_total(kfold_x_test,trained_weights)

                    # Calculate the indexes and store them
                    acc.append(perceptron_accuracy(kfold_y_test,predicted))
                    kappa.append(cohen_kappa_score(kfold_y_test, predicted))
                    f1.append(fbeta_score(kfold_y_test, predicted, beta=1))

                print("Testing the model with step = ", step)
                print("Testing the model with number of iterations = ", n_iter)
                
                # Store the mean of the indexes for the 4 folds
                step_full.append(step)
                n_iter_full.append(n_iter)
                acc_full.append(np.mean(acc))
                kappa_full.append(np.mean(kappa))
                f1_full.append(np.mean(f1))


        # Create pandas dataset and store it in a csv
        dic = {'Step':step_full,'N_Iter':n_iter_full,'Accuracy':acc_full,'Kappa':kappa_full,'F1':f1_full}
        df_grid_search = pd.DataFrame(dic)
        df_grid_search.to_csv('Results/grid_search_perceptron.csv')
        print("Tuning Ready!")

    else:
        # In case the parameters were already tuned
        df_grid_search = pd.read_csv('Results/grid_search_perceptron.csv')
        print("Previous tuning detected, skipping tuning")

    # Search the bigger F1 index in the dataframe
    row_max = df_grid_search['F1'].argmax()

    # Get the the better performing step and number of iterations
    step_max = float(df_grid_search['Step'].values[row_max])
    n_iter_max = int(df_grid_search['N_Iter'].values[row_max])

    print("The parameters were chosen looking at the maximum F1 score")
    print("F1 score:", df_grid_search['F1'][row_max])
    print("Accuracy:", df_grid_search['Accuracy'][row_max])
    print("Cohen's Kappa:", df_grid_search['Kappa'][row_max])
    print("Using:", df_grid_search['N_Iter'][row_max], "epochs with step = ", df_grid_search['Step'][row_max])

    print("Training final model")

    # Training of the final model
    final_weights = perceptron_train(x_train,y_train,step = step_max,n_iter = n_iter_max)

    print("Weights:", final_weights)

    predicted_final = perceptron_prediction_total(x_test,final_weights)

    # Calculate the indexes of the final results
    acc_final = perceptron_accuracy(y_test,predicted_final)
    kappa_final = cohen_kappa_score(y_test,predicted_final)
    F1_final = fbeta_score(y_test,predicted_final, beta=1)

    print("Testing final model")
    print("F1 score:", F1_final)
    print("Accuracy:", acc_final)
    print("Cohen's Kappa:", kappa_final)

    # Save the final results
    dic = {'Step':step_max,'N_Iter':n_iter_max,'Accuracy':acc_final,'Kappa':kappa_final,'F1':F1_final}
    df_results = pd.DataFrame(dic, index=[0])
    df_results.to_csv('Results/results_perceptron.csv')


if __name__ == '__main__':
	main()





