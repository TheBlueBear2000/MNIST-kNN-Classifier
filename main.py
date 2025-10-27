from torchvision import datasets
from PIL import Image, ImageFont, ImageDraw
from random import randint, seed
from math import sqrt

SAMPLE_SIZE = 60000
INPUT_SIZE = 6
K = 11

# Seed dataset selection, so that same model could be generated twice later
#SEED = randint(0, 100000)
SEED = 500
print ("Sample set seed:", SEED)
seed(SEED) 


# MNIST is loaded as a list of 60,000 tuples, which contain a 28x28 PIL image paired with the number it represents
# Load training data
train_dataset = datasets.MNIST(root='./data', train=True, download=True)
# Load testing data
input_dataset = datasets.MNIST(root='./data', train=False, download=True)


# NOTE: not accounting for repeats
sampled_dataset = []
for i in range(SAMPLE_SIZE):
    sampled_dataset.append(train_dataset[randint(0, len(train_dataset)-1)])


def eudlideanDistance(image1_pix, image2_pix, imageDimensions=(28, 28)):
    squared_total = 0
    for y in range(imageDimensions[1]):
        for x in range(imageDimensions[0]):
            squared_total += (image1_pix[x, y] - image2_pix[x, y]) ** 2
    #print("squared total:", squared_total)
    return sqrt(squared_total)

def checkKNearestNeighbours(input_data, dataset, k=11):
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
    
    return max(set(neighbours['classes']), key=neighbours['classes'].count)

# Picks the first INPUT_SIZE number of items from input data, so same data will always be tested
# for i in range(INPUT_SIZE):
#     print(f"Real value of item {i}:", input_dataset[i][1])
#     print(f"Estimated value of item {i}:", checkKNearestNeighbours(input_dataset[i][0], sampled_dataset))

outputImage = Image.new(mode="RGB", size=(600, 400))
for i in range(INPUT_SIZE):
    outputImage.paste(input_dataset[i][0].resize((180, 180), resample = Image.Resampling.NEAREST), ((200 * (i % 3)) + 10, 200 * (i // 3)))
    draw = ImageDraw.Draw(outputImage)
    draw.text(((200 * (i % 3)) + 10, (200 * (i // 3)) + 160), f"Real value of item: {input_dataset[i][1]}\nEstimated value of item: {checkKNearestNeighbours(input_dataset[i][0], sampled_dataset)}", fill=(255, 0, 0))
outputImage.show()