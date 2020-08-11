###################################
# Julian Cabezas Pena
# Deep Learning Fundamentals
# University of Adelaide
# Assingment 1
# Random forest algorithm testiong in the PIMA diabetes data
####################################


import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, fbeta_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

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

# Main function, it will preprocess the data, create train and test data,
# tune the parameters of the Random forest and evaluate the final model
def main():

    # Read the database using pandas
    diabetes = pd.read_csv("Input_Data/datasets_228_482_diabetes.csv")

    # Recode the output column to get -1 and 1 output values
    diabetes['Outcome'] = np.where(diabetes['Outcome'] == 0, -1, diabetes['Outcome'])

    # Scale the variables between -1 and +1
    diabetes['Pregnancies'] = scale_data(diabetes['Pregnancies'],-1,1)
    diabetes['Glucose'] = scale_data(diabetes['Glucose'],-1,1)
    diabetes['BloodPressure'] = scale_data(diabetes['BloodPressure'],-1,1)
    diabetes['SkinThickness'] = scale_data(diabetes['SkinThickness'],-1,1)
    diabetes['Insulin'] = scale_data(diabetes['Insulin'],-1,1)
    diabetes['BMI'] = scale_data(diabetes['BMI'],-1,1)
    diabetes['DiabetesPedigreeFunction'] = scale_data(diabetes['DiabetesPedigreeFunction'],-1,1)
    diabetes['Age'] = scale_data(diabetes['Age'],-1,1)

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
    if not os.path.exists('Results/grid_search_random_forest.csv'):

        print("Parameter tuning not detected, tuning step and number of epochs parameters")

        # set up a tratified 4-fold partition of the train data
        k_fold_strat = StratifiedKFold(n_splits=4, random_state=23,shuffle=True)

        # Test different maximum number of features in each split
        maxf_array = np.arange(start=1, stop=9, step=1)

        # Test different numbers of trees from 10 to 550
        iter_array = np.arange(start=10, stop=510, step=50)


        # Store the partial results in lists
        maxf_full = []
        n_trees_full = []
        acc_full = []
        kappa_full = []
        f1_full = []

        # Loop trough the different combinations of step and number of iterations
        for maxf in maxf_array:

            for n_trees in iter_array:
                
                # Store partial results for accuracy, cohen kappa and F1
                acc = []
                kappa = []
                f1 = []

                # Initialize the random forest classifier
                rf = RandomForestClassifier(n_estimators=n_trees, max_features = maxf ,random_state=124)

                # Iterate thorgh the 4 folds
                for kfold_train_index, kfold_test_index in k_fold_strat.split(x_train, y_train):
                    
                    # Get the split into train and test
                    kfold_x_train, kfold_x_test = x[kfold_train_index][:], x[kfold_test_index][:]
                    kfold_y_train, kfold_y_test = y[kfold_train_index], y[kfold_test_index]

                    # Train the RF and get the predicted values
                    rf.fit(kfold_x_train,kfold_y_train)
                    predicted = rf.predict(kfold_x_test)

                    # Calculate the indexes and store them
                    acc.append(accuracy_score(kfold_y_test,predicted))
                    kappa.append(cohen_kappa_score(kfold_y_test, predicted))
                    f1.append(fbeta_score(kfold_y_test, predicted, beta=1))

                print("Testing the model with maximum number of features = ", maxf)
                print("Testing the model with number of trees = ", n_trees)
                
                # Store the mean of the indexes for the 4 folds
                maxf_full.append(maxf)
                n_trees_full.append(n_trees)
                acc_full.append(np.mean(acc))
                kappa_full.append(np.mean(kappa))
                f1_full.append(np.mean(f1))

        # Create pandas dataset and store it in a csv
        dic = {'maxf':maxf_full,'n_trees':n_trees_full,'Accuracy':acc_full,'Kappa':kappa_full,'F1':f1_full}
        df_grid_search = pd.DataFrame(dic)
        df_grid_search.to_csv('Results/grid_search_random_forest.csv')
        print("Tuning Ready!")

    else:
        # In case the parameters were already tuned
        df_grid_search = pd.read_csv('Results/grid_search_random_forest.csv')
        print("Previous tuning detected, skipping tuning")


    # Search the bigger F1 index in the dataframe
    row_max = df_grid_search['F1'].argmax()

    # Get the the better performing step and number of iterations
    maxf_max = int(df_grid_search['maxf'].values[row_max])
    n_trees_max = int(df_grid_search['n_trees'].values[row_max])

    print("The parameters were chosen looking at the maximum F1 score")
    print("F1 score:", df_grid_search['F1'][row_max])
    print("Accuracy:", df_grid_search['Accuracy'][row_max])
    print("Cohen's Kappa:", df_grid_search['Kappa'][row_max])
    print("Using:", df_grid_search['maxf'][row_max], "as maximum number of features and ", df_grid_search['n_trees'][row_max], "trees")

    print("Training final model")

    # Training of the final model
    rf = RandomForestClassifier(n_estimators=n_trees_max, max_features = maxf_max ,random_state=124)

    rf.fit(x_train,y_train)
    predicted_final = rf.predict(x_test)

    # Calculate the indexes of the final results
    acc_final = accuracy_score(y_test,predicted_final)
    kappa_final = cohen_kappa_score(y_test,predicted_final)
    F1_final = fbeta_score(y_test,predicted_final, beta=1)

    print("Testing final model")
    print("F1 score:", F1_final)
    print("Accuracy:", acc_final)
    print("Cohen's Kappa:", kappa_final)

    # Save the final results
    dic = {'maxf':maxf_max,'n_trees':n_trees_max,'Accuracy':acc_final,'Kappa':kappa_final,'F1':F1_final}
    df_results = pd.DataFrame(dic, index=[0])
    df_results.to_csv('Results/results_random_forest.csv')



if __name__ == '__main__':
	main()





