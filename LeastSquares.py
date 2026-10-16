from implementations import least_squares
from helpers import *
#import os
def main():
    x_train, x_test, y_train, train_ids, test_ids=load_csv_data("data/dataset/")
    
    """"Handle missing values by replacing them with the mean of the corresponding feature."""
    medie = np.nanmean(x_train, axis=0)
    righe, colonne = np.where(np.isnan(x_train))
    x_train[righe, colonne] = medie[colonne]
    righe, colonne = np.where(np.isnan(x_test))
    x_test[righe, colonne] = medie[colonne]

    """
    # Testing the least_squares function with a validation set
    lunghezza = len(x_train)
    x_allenamento = x_train[:lunghezza//2]
    x_prova = x_train[lunghezza//2:]

    y_allenamento = y_train[:lunghezza//2]
    y_prova = y_train[lunghezza//2:]

    

    if (lunghezza-lunghezza//2)==1:
        x_prova = x_prova[:-1]
        y_prova = y_prova[:-1]
    else:
        x_allenamento = x_allenamento[:-1]
        y_allenamento = y_allenamento[:-1]
    w, loss = least_squares(y_allenamento, x_allenamento)
    y_previsione = x_prova @ w
    mse = np.mean((y_prova - y_previsione) ** 2)
    print("MSE on validation set:", mse)
    print("y_previsione:", y_previsione)
    print("y_prova:", y_prova)"""
    
    w, loss = least_squares(y_train, x_train)
    y_pred = x_test @ w
    for i in range(len(y_pred)):
        if y_pred[i] < 0:
            y_pred[i] = -1
        else:
            y_pred[i] = 1
    create_csv_submission(test_ids, y_pred, "least_squares_0.csv")

if __name__ == "__main__":
    main()