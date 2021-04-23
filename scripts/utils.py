from sklearn.model_selection import train_test_split

def get_train_test_valid_split(X, y, valid=0.2, test=0.2):

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size = test, random_state=0)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size = valid/(1-test), random_state=0)

    return X_train, X_valid, X_test, y_train, y_valid, y_test