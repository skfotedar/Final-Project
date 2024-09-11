import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from PIL import Image

# list of images
image_path = [  'GL_vel_mosaic_Annual_01Dec14_30Nov15_browse_v05.0.jpg',
                'GL_vel_mosaic_Annual_01Dec15_30Nov16_browse_v05.0.jpg',
                'GL_vel_mosaic_Annual_01Dec16_30Nov17_browse_v05.0.jpg',
                'GL_vel_mosaic_Annual_01Dec17_30Nov18_browse_v05.0.jpg',
                'GL_vel_mosaic_Annual_01Dec18_30Nov19_browse_v05.0.jpg',
                'GL_vel_mosaic_Annual_01Dec19_30Nov20_browse_v05.0.jpg',
                'GL_vel_mosaic_Annual_01Dec20_30Nov21_browse_v05.0.jpg']

resized_images = []
#reduce to 10% of the original size
for path in image_path:
    img = load_img(path)
    img = img.resize((303, 548))
    resized_images.append(img)

# load the resized images and normalize
images = []
for path in resized_images:
    img = path
    img_array = img_to_array(img)
    images.append(img_array)
images = np.array(images) / 255.0

X = np.expand_dims(images, axis=0)  # Add batch dimension
y = np.expand_dims(images[-1], axis=0)  # The last image as the target

#create the model
model = Sequential()
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(None, 548, 303, 3)))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(303 * 548 * 3, activation='sigmoid'))
model.add(tf.keras.layers.Reshape((548, 303, 3)))
print(model.summary())

model.compile(optimizer='adam', loss='mse')


model.fit(X, y, epochs=10, verbose=1)

predicted_image = model.predict(X)
predicted_image = np.squeeze(predicted_image, axis=0)  # Remove batch dimension

plt.imshow(predicted_image)
plt.axis('off')
plt.show()

predicted_image = (predicted_image * 255).astype(np.uint8)
tf.keras.preprocessing.image.save_img('predicted_image.jpg', predicted_image)

loss = model.evaluate(X, y)
print(f'Model Loss: {loss}')