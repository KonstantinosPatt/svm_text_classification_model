# from sklearn.model_selection import cross_val_score
from c_svm_model_train import create_svm_model
from d_svm_model_test import test_svm_model
from itertools import combinations

def cross_validation(folder, array, train_csv, train_folds, percentile):
    num_of_parts = len(array)
    test_folds = num_of_parts - train_folds
    avg_accuracy = 0.0
    test_combinations = list(combinations(array, test_folds))

    for combination in test_combinations:
        test_parts = list(combination)
        train_array = [part for part in array if part not in test_parts]

        #print(test_parts, train_array)

        create_svm_model(folder, train_array, percentile, None)
        accuracy = test_svm_model(folder, test_parts, percentile=100, train_csv=train_csv)
        avg_accuracy += accuracy

    avg_accuracy = avg_accuracy/len(test_combinations)
    return avg_accuracy


if __name__ == '__main__':
    array = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    folder = "./output"
    train_csv = "train_one_hot.csv"

    ## Step 5
    # 10-fold cross validation with k=10 and k-1=9 training folds
    #avg_accuracy = cross_validation(folder, array, train_csv, len(array)-1, 100)
    #print("Average accuracy, 9 training, 1 testing folds:", avg_accuracy)

    # Cross validation with 3 training folds
    #avg_accuracy = cross_validation(folder, array, train_csv, 3, 100)
    #print("Average accuracy, 3 training, 7 testing folds:", avg_accuracy)

    # Cross validation with 6 training folds
    #avg_accuracy = cross_validation(folder, array, train_csv, 6, 100)
    #print("Average accuracy, 6 training, 4 testing folds:", avg_accuracy)

    ## Step 6
    # Feature selection, keep 50% of most important features
    avg_accuracy = cross_validation(folder, array, train_csv, len(array)-1, 10)
    print("Average accuracy, 9 training, 1 testing folds, 10%:", avg_accuracy)

    avg_accuracy = cross_validation(folder, array, train_csv, len(array)-1, 40)
    print("Average accuracy, 9 training, 1 testing folds, 40%:", avg_accuracy)

    avg_accuracy = cross_validation(folder, array, train_csv, len(array)-1, 80)
    print("Average accuracy, 9 training, 1 testing folds, 80%:", avg_accuracy)