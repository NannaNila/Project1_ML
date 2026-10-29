from implementations import least_squares
from helpers import *
#import os
def main():
    x_train, x_test, y_train, train_ids, test_ids=load_csv_data("data/dataset/")
    
    #Handle missing values by replacing them with the median of the corresponding feature.
    median = np.nanmedian(x_train, axis=0)
    row, col = np.where(np.isnan(x_train))
    x_train[row, col] = median[col]

    median = np.nanmedian(x_test, axis=0)
    row, col = np.where(np.isnan(x_test))
    x_test[row, col] = median[col]

    # Training
    w, loss = least_squares(y_train, x_train)
    y_pred = x_test @ w
    # Post-processing predictions to be either -1 or 1
    for i in range(len(y_pred)):
        if y_pred[i] < 0:
            y_pred[i] = -1
        else:
            y_pred[i] = 1
    create_csv_submission(test_ids, y_pred, "least_squares_0.csv")

if __name__ == "__main__":
    main()