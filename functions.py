from torchvision import datasets
from random import randint
from math import sqrt

SAMPLE_SIZE = 5000

def eudlideanDistance(image1_pix, image2_pix, imageDimensions=(28, 28)):
    squared_total = 0
    for y in range(imageDimensions[1]):
        for x in range(imageDimensions[0]):
            squared_total += (image1_pix[x, y] - image2_pix[x, y]) ** 2
    #print("squared total:", squared_total)
    return sqrt(squared_total)

def getDataset():
    train_dataset = datasets.MNIST(root='./data', train=True, download=True)
    sampled_dataset = []
    for i in range(SAMPLE_SIZE):
        sampled_dataset.append(train_dataset[randint(0, len(train_dataset)-1)])
    return sampled_dataset

def checkKNearestNeighbours(input_data, dataset = getDataset(), k=11):
    neighbours = {
        "distances": [],
        "classes": []
    }
    input_data_pix = input_data.load()
    for item in dataset:
        distance = eudlideanDistance(input_data_pix, item[0].load())
        # If we have not yet found k neighbours
        if len(neighbours['distances']) < k:
            neighbours['distances'].append(distance)
            neighbours['classes'].append(item[1])
        elif distance < max(neighbours['distances']):
            insert_index = neighbours['distances'].index(max(neighbours['distances']))
            neighbours['distances'][insert_index] = distance
            neighbours['classes'][insert_index] = item[1]
    
    output = []
    for item in set(neighbours['classes']):
        output.append((item, neighbours['classes'].count(item)))
    return sorted(output, key=lambda x: x[1], reverse=True)
