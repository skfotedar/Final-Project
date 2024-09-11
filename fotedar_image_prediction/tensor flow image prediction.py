import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from PIL import Image

# list of images
image_path = ['N_197901_anom_hires_v3.0.png', 'N_198001_anom_hires_v3.0.png',
            'N_198101_anom_hires_v3.0.png', 'N_198201_anom_hires_v3.0.png',
            'N_198301_anom_hires_v3.0.png']

# Define the size of the crop
crop_width, crop_height = 800, 800

#crop images
cropped_images = []
for path in image_path:
    img = load_img(path)
    # Calculate the coordinates for the crop box
    left = (img.width - crop_width) / 2
    top = (img.height - crop_height) / 2
    right = (img.width + crop_width) / 2
    bottom = (img.height + crop_height) / 2
    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))
    cropped_images.append(cropped_img)

# load the cropped images
images = []
for path in cropped_images:
    img = path
    img_array = img_to_array(img)
    images.append(img_array)
images = np.array(images) / 255.0

X = np.expand_dims(images, axis=0)  # Add batch dimension
y = np.expand_dims(images[-1], axis=0)  # The last image as the target

#create the model
model = Sequential()
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(None, 800, 800, 3)))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(800 * 800 * 3, activation='sigmoid'))
model.add(tf.keras.layers.Reshape((800, 800, 3)))

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