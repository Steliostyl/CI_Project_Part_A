import numpy as np
from sklearn.datasets import load_iris

def parse_data(file_name):
    paragraph = []
    with open(file_name, "r") as file:
        for line in file:
            # Create an 8520 long array filled with 0
            word_count = np.zeros(8520, dtype=np.float64)
            # From the input file, extract only the numbers and add them to an array
            numbers = [int(number) for number in line.split() if number.isdigit()]
            for n in numbers:
                word_count[n] += 1
            paragraph.append(word_count)
    return centerArray(np.array(paragraph))
    #return normalizeArray(np.array(paragraph))
    #return standardizeArray(np.array(paragraph))
    #return (np.array(paragraph))

def parse_label(file_name):
    categories = []
    with open(file_name, "r") as file:
        for line in file:
            num_line = np.array([int(number) for number in line.split()])
            categories.append(num_line)
    return np.array(categories)

def centerArray(np_array):
    return np_array - np_array.mean()

def normalizeArray(np_array):
    max = np.max(np_array)
    return np_array/max    

def standardizeArray(np_array):
    return centerArray(normalizeArray(np_array))