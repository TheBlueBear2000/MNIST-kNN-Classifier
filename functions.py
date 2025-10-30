from torchvision import datasets
from random import randint
from math import sqrt
from helpers import printProgressBar
from time import process_time

SAMPLE_SIZE = 5000

def eudlideanDistance(image1_pix, image2_pix, imageDimensions=(28, 28)):
    squared_total = 0
    for y in range(imageDimensions[1]):
        for x in range(imageDimensions[0]):
            squared_total += (image1_pix[x, y] - image2_pix[x, y]) ** 2
    #print("squared total:", squared_total)
    return sqrt(squared_total)

def getDataset(reduced_sample=True, new_sample_size=SAMPLE_SIZE):
    train_dataset = datasets.MNIST(root='./data', train=True, download=True)
    if not reduced_sample:
        return list(train_dataset)
    sampled_dataset = []
    for i in range(new_sample_size):
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


def hartAlgorithm(inputDataset):
    start_time = process_time()
    output_dataset = []
    cleared_pass = False
    passes = 0
    while not cleared_pass:
        passes += 1
        cleared_pass = True
        for i, item in enumerate(inputDataset):
            if item != None:
                if i%20 == 0:
                    printProgressBar(i, len(inputDataset), prefix = f'Pass {passes}:', suffix = 'Complete')
                predicted_states = checkKNearestNeighbours(item[0], output_dataset)
                predicted_state = predicted_states[0] if len(predicted_states) > 0 else 10 # 10 is unobtainable, so the item will be added regardless if the output is empty
                if predicted_state != item[1]:
                    output_dataset.append(item)
                    inputDataset[i] = None
                    cleared_pass = False
        print(f"Condensed dataset length is {len(output_dataset)}\nProcessing took {process_time() - start_time} seconds\n")
    return output_dataset