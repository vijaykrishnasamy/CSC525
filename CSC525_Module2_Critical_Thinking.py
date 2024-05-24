import csv
import math
from collections import Counter

def load_data(filename):
    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            data.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), row[4]])
    return data

def euclidean_distance(point1, point2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

def get_neighbors(training_data, test_point, k):
    distances = []
    for train_point in training_data:
        distance = euclidean_distance(train_point[:-1], test_point)
        distances.append((train_point, distance))
    distances.sort(key=lambda x: x[1])
    neighbors = [dist[0] for dist in distances[:k]]
    return neighbors

def predict_classification(training_data, test_point, k):
    neighbors = get_neighbors(training_data, test_point, k)
    output_values = [neighbor[-1] for neighbor in neighbors]
    prediction = Counter(output_values).most_common(1)[0][0]
    return prediction

def main():
    # Load data from CSV
    filename = 'iris.csv'
    data = load_data(filename)
    
    k = 3
    
    # four floating point numbers representing sepal length, sepal width, petal length, and petal width
    sepal_length = float(input("Enter sepal length (cm): "))
    sepal_width = float(input("Enter sepal width (cm): "))
    petal_length = float(input("Enter petal length (cm): "))
    petal_width = float(input("Enter petal width (cm): "))
    
    test_point = [sepal_length, sepal_width, petal_length, petal_width]
    
    # Predict the class
    prediction = predict_classification(data, test_point, k)
    print(f"The predicted class for the given measurements is: {prediction}")

if __name__ == '__main__':
    main()
