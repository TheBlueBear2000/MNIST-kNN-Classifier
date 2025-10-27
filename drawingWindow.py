import pygame
from time import sleep
import numpy
from functions import checkKNearestNeighbours
from PIL import Image

# initiate pygame and give permission
# to use pygame's functionality.
pygame.init()

WINDOW_SIZE = (600,800)
CANVAS_SIZE = 600
GAUSIAN_KERNAL_SIZE = 5
PEN_WEIGHT = 0.8

mousedown = False
#painting_distribution = random.normal(size=(GAUSIAN_KERNAL_SIZE, GAUSIAN_KERNAL_SIZE))
painting_distribution = [[ 0.00730688,  0.03274718,  0.05399097,  0.03274718,  0.00730688],
                        [ 0.03274718,  0.14676266,  0.24197072,  0.14676266,  0.03274718],
                        [ 0.05399097,  0.24197072,  0.39894228,  0.24197072,  0.05399097],
                        [ 0.03274718,  0.14676266,  0.24197072,  0.14676266,  0.03274718],
                        [ 0.00730688,  0.03274718,  0.05399097,  0.03274718,  0.00730688]]
font = pygame.font.Font('freesansbold.ttf', 32)

# create the display surface object
# of specific dimension.
window = pygame.display.set_mode(WINDOW_SIZE)

# Fill the scree with white color
window.fill((255, 255, 255))

# Using draw.rect module of
# pygame to draw the outlined rectangle
square_size = CANVAS_SIZE/28


def addPoint(image, point_loc):
    image[point_loc[1]][point_loc[0]] = 1
    return image
    
def applyGaussian(image):
    output = createBlankImage()
    for y in range(28):
        for x in range(28):
            if image[y][x] == 1:
                for rel_y in range(GAUSIAN_KERNAL_SIZE):
                    for rel_x in range(GAUSIAN_KERNAL_SIZE):
                        new_x = x - (GAUSIAN_KERNAL_SIZE // 2) + rel_x
                        new_y = y - (GAUSIAN_KERNAL_SIZE // 2) + rel_y
                        if (new_x >= 0) & (new_x < 28) & (new_y >= 0) & (new_y < 28):
                            output[new_y][new_x] += 255 * painting_distribution[rel_y][rel_x] * (PEN_WEIGHT/painting_distribution[len(painting_distribution)//2][len(painting_distribution[0])//2])
                            if output[new_y][new_x] > 255:
                                output[new_y][new_x] = 255
    return output


def renderSpace(image):
    for y in range(28):
        for x in range(28):
            pygame.draw.rect(window, (image[y][x], image[y][x], image[y][x]), 
                            [(square_size * x) + 1, (square_size * y) + 1, square_size-1, square_size-1], 0)

def createBlankImage():
    image = []
    for y in range(28):
        row = []
        for x in range(28):
            row.append(0)
        image.append(row)
    return image

def defineClass(image):
    pil_image = Image.fromarray(numpy.array(image, dtype=numpy.uint8))
    determined_number = checkKNearestNeighbours(pil_image, k=23)
    
    pygame.draw.rect(window, (255, 255, 255), [0, CANVAS_SIZE, CANVAS_SIZE, WINDOW_SIZE[1]-CANVAS_SIZE])
    text = font.render(f"Num: {determined_number}", True, (255,0,0))
    window.blit(text, (0, CANVAS_SIZE))


image = createBlankImage()
output_image = createBlankImage()
draw_history = []

while True:
    # Events
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                draw_history.insert(0, [])
                mousedown = True
            
            elif event.type == pygame.MOUSEBUTTONUP:
                mousedown = False
                output_image = applyGaussian(image)
                defineClass(output_image)
            
            elif (event.type == pygame.MOUSEMOTION) & mousedown:
                rel_x, rel_y = pygame.mouse.get_pos()
                rel_x = int(rel_x // square_size)
                rel_y = int(rel_y // square_size)
                
                if (rel_x >= 0) & (rel_x < 28) & (rel_y >= 0) & (rel_y < 28):
                    draw_history[0].append((rel_x, rel_y))
                    image = addPoint(image, (rel_x, rel_y))
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_z:
                    keys = pygame.key.get_pressed()
                    if (keys[pygame.K_LMETA] or keys[pygame.K_RMETA]) and len(draw_history) > 0:
                        for item in draw_history[0]:
                            image[item[1]][item[0]] = 0
                        draw_history.pop(0)
                        output_image = applyGaussian(image)
                        defineClass(output_image)
    
    output_image = applyGaussian(image)
    renderSpace(output_image)
    # Draws the surface object to the screen.
    pygame.display.update()
    
    
    sleep (0.01)