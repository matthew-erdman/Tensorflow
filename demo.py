import tensorflow.keras.models as models
import tensorflow.keras.utils as utils
import tensorflow as tf
from random import randrange, choice
from pickle import load
from graph import graph

model = models.load_model('rice_model_save')
with open('rice_model_save/class_names.data', 'rb') as f:
    class_names = load(f)

print('\n---------------------')

image_count = 100
results = [[], []]
incorrect = {}
for i in range(image_count):
    rice = choice(class_names)
    image_num = randrange(2000, 10000)  # select images outside of train/test set
    image = utils.load_img(f'all_images/{rice}/{rice} ({image_num}).jpg')
    image_array = utils.img_to_array(image)[tf.newaxis, ...]
    predictions = list(model(image_array).numpy()[0])
    results[0].append(rice)
    results[1].append(class_names[predictions.index(max(predictions))])
    # track incorrect predictions
    truth = results[0][i]
    guess = results[1][i]
    if guess != truth:
        print(f'Incorrectly guessed {guess} when {truth} was correct')
        incorrect[truth] = incorrect.get(truth, 0) + 1

correct_count = image_count - sum(incorrect.values())
print(f'\nTested {image_count} images and correctly predicted {correct_count} ({correct_count/image_count:.0%})')
print(f'Most often missed: {max(incorrect, key=incorrect.get)} ({max(incorrect.values())} times, {max(incorrect.values()) / sum(incorrect.values()):.0%} of all missed)')
input('...')
graph()