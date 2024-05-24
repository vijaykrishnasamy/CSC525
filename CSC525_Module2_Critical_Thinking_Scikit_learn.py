import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import joblib

def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y, iris

def train_knn(X, y, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    return knn

def main():
    # Load data
    X, y, iris = load_data()
    
    k = 3
    
    # Train the KNN classifier
    knn = train_knn(X, y, k)
    
    # Save the model to disk
    joblib.dump(knn, 'knn_model.joblib')
    
    # four floating point numbers representing sepal length, sepal width, petal length, and petal width
    sepal_length = float(input("Enter sepal length (cm): "))
    sepal_width = float(input("Enter sepal width (cm): "))
    petal_length = float(input("Enter petal length (cm): "))
    petal_width = float(input("Enter petal width (cm): "))
    
    # Test point
    test_point = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Predict the class
    prediction = knn.predict(test_point)
    class_name = iris.target_names[prediction[0]]
    print(f"The predicted class for the given measurements is: {class_name}")

if __name__ == '__main__':
    main()
