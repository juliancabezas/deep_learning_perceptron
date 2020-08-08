import numpy as np
import pandas as pd

# Predict the output for a single sample
def perceptron_prediction(input,w) {

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

}

