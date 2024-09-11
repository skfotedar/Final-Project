import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 400 * 400, 3 * 800 * 800)


# list of images
image_path = ['N_197901_anom_hires_v3.0.png', 'N_198001_anom_hires_v3.0.png',
            'N_198101_anom_hires_v3.0.png', 'N_198201_anom_hires_v3.0.png',
            'N_198301_anom_hires_v3.0.png']

#crop images
cropped_images = []
crop_size = 800
for images in image_path:
    image = Image.open(images).convert('RGB')
    width, height = image.size
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2
    image = image.crop((left, top, right, bottom))
    cropped_images.append(image)

#read in cropped images and convert to number array and scale
list_of_images = []
for images in cropped_images:
    image = transforms.ToTensor()(images)
    list_of_images.append(image)

#stack the tensors
stacked_images = torch.stack(list_of_images)
print(stacked_images.shape)

# Initialize the model, loss function, and optimizer
model = SimpleCNN()
#criterion = nn.MSELoss()
#optimizer = optim.Adam(model.parameters(), lr=0.001)






