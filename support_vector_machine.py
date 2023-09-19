from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

model = SVC(kernel = 'linear')

def train_model(X_train, y_train):
    model.fit(X_train, y_train)


def prediction(X_test):
    y_pred = model.predict(X_test)
    return y_pred

def looking_at_accuracy(X_test,y_test):
    y_pred = prediction(X_test)
    cm = confusion_matrix(y_test,y_pred)
    return cm, accuracy_score(y_test, y_pred)







