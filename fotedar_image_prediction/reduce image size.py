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
i=0
for path in image_path:
    i+=1
    img = load_img(path)
    img = img.resize((303, 548))
    img.save('img'+str(i)+'.jpg')
