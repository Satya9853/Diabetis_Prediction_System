# Importing The Dependencies

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import seaborn as sns
from IPython.display import clear_output
import time
import sys

# Data Collection -> PIMA Diabetic data(Female)

diabetis_df = pd.read_csv("./diabetes.csv")

diabetis_df.head()

diabetis_df.shape

diabetis_df.describe()

diabetis_df["Outcome"].value_counts()

# 0 -> Non Diabetic People 1 -> Diabetic People

diabetis_df.groupby("Outcome").mean()
diabetis_df.isnull().any()

# Data Preprocessing

# Separating the data and the labels
data = diabetis_df.drop(axis=1, columns="Outcome")
label = diabetis_df["Outcome"]

# Data Standardization using Standard Scaler
scaler = StandardScaler()
scaler.fit(data)
standardized_data = scaler.transform(data)

print(standardized_data)

# Splitting the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(standardized_data,
                                                    label,
                                                    random_state=2,
                                                    test_size=0.2,
                                                    stratify=label
                                                    )

print(standardized_data.shape, X_train.shape, X_test.shape)

# Training the Model Using Support Vector Machine

linear_svc = svm.SVC(kernel="linear")
radial_svc = svm.SVC(kernel="rbf")
polynomial_svc = svm.SVC(kernel="poly")

linear_svc.fit(X_train, y_train)
radial_svc.fit(X_train, y_train)
polynomial_svc.fit(X_train, y_train)

print(f"kernel = Linear \nTraining = {linear_svc.score(X_train, y_train)}")
print(f"Test = {linear_svc.score(X_test, y_test)}")

print(f"kernel = Radial \nTraining = {radial_svc.score(X_train, y_train)}")
print(f"Test = {radial_svc.score(X_test, y_test)}")

print(f"kernel = Polynomial \nTraining = {polynomial_svc.score(X_train, y_train)}")
print(f"Test = {polynomial_svc.score(X_test, y_test)}")

# Using Accuracy to Analyse the Model

X_train_accuracy = linear_svc.predict(X_train)
X_train_score = accuracy_score(X_train_accuracy, y_train)
print(X_train_score)

X_test_accuracy = linear_svc.predict(X_test)
X_test_score = accuracy_score(X_test_accuracy, y_test)
print(X_test_score)

# Diabetis Prediction System


class Timer:
    def __init__(self, seconds):
        self.seconds = seconds

    def countdown(self):
        total_seconds = self.seconds
        sys.stdout.write("{:2d} seconds remaining.".format(total_seconds))
        while (total_seconds > 0):
            time.sleep(1)
            total_seconds -= 1
            sys.stdout.write("\r")
            sys.stdout.write("{:2d} seconds remaining.".format(total_seconds))
            sys.stdout.flush()


details = ["Pregnancies", "Glucose",
           "BloodPressure", "SkinThickness",
           "Insulin", "BMI",
           "DiabetesPedigreeFunction", "Age"]

values = []
my_timer = Timer(seconds=5)
while True:
    print("Please Provide the details to predict Diabetis\n")
    for i in details:
        input_data = float(input(f"{i} = "))
        values.append(input_data)

    values_as_numpy_array = np.asarray(values)
    data = values_as_numpy_array.reshape(1, -1)
    scaled_data = scaler.transform(data)
    prediction = linear_svc.predict(scaled_data)
    if prediction[0] == 0:
        print("Congratulations you are not having diabetis")
    else:
        print("Sorry you are having diabetis")
    my_timer.countdown()
    clear_output()
