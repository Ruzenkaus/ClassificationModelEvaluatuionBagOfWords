import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import k_nearest_neighbors, logistic_regression,Decision_tree_classification,Naive_Bayes, support_vector_machine, random_forest_classification
import matplotlib.pyplot as plt
import data_preprocessing  # Make sure you have this module if it's custom
from sklearn import metrics
import kernel_svm
# Load the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
corpus = data_preprocessing.create_corpus(dataset)

# Create bag_of_words
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.2)

# Train models
Naive_Bayes.train_model(X_train, y_train)
Decision_tree_classification.train_model(X_train, y_train)
k_nearest_neighbors.train_model(X_train, y_train)
kernel_svm.train_model(X_train, y_train)
logistic_regression.train_model(X_train, y_train)
support_vector_machine.train_model(X_train, y_train)
random_forest_classification.train_model(X_train, y_train)
# Get accuracy and confusion matrices
nb_cm, nb_accuracy = Naive_Bayes.looking_at_accuracy(X_test, y_test)
dt_cm, dt_accuracy = Decision_tree_classification.looking_at_accuracy(X_test, y_test)
k_cm, k_accuracy = k_nearest_neighbors.looking_at_accuracy(X_test, y_test)
ks_cm, ks_accuracy = kernel_svm.looking_at_accuracy(X_test, y_test)
lr_cm, lr_accuracy = logistic_regression.looking_at_accuracy(X_test, y_test)
svm_cm, svm_accuracy = support_vector_machine.looking_at_accuracy(X_test, y_test)
rfc_cm, rfc_accuracy = support_vector_machine.looking_at_accuracy(X_test, y_test)
cms = [nb_cm, dt_cm, k_cm, ks_cm, lr_cm, svm_cm, rfc_cm]
accuracies = [nb_accuracy, dt_accuracy, k_accuracy, ks_accuracy, lr_accuracy, svm_accuracy, rfc_accuracy]
names = ['Naive Bayes', 'Desicison Tree', 'K-nearest', 'Kernel SVM', 'Logistic Regression', 'SVM', 'Random Forest']
# Create subplots
figure, axes = plt.subplots(1, len(cms), figsize=(15, 4))

for i in range(len(cms)):
    disp_cm = metrics.ConfusionMatrixDisplay(confusion_matrix=cms[i], display_labels=['True', 'False'])
    disp_cm.plot(ax=axes[i], xticks_rotation='vertical', values_format="d")
    axes[i].set_title(f"{names[i]} Accuracy: {accuracies[i]:.2f}")

plt.tight_layout()
plt.show()
