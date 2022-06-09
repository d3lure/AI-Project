import time, os, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

img_height, img_width = 150, 150
class_names = ['cat', 'dog']
model = keras.models.load_model('cnn_model.h5')

img_dir = "../imgs/" + random.choice(os.listdir("../imgs/"))

img = tf.keras.utils.load_img(img_dir, target_size=(img_height, img_width))
plt.imshow(img)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

start = time.time()
predictions = model(img_array)
end = time.time()

score = tf.nn.softmax(predictions[0])

plt.xlabel("Prediction time: {:.2f}[s]".format(end-start))
plt.title("This image is most likely a {} with a {:.2f}% confidence."
          .format(class_names[np.argmax(score)], 100 * np.max(score)))
plt.show()
