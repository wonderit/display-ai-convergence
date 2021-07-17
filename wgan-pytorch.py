#!/usr/bin/env python
# coding: utf-8

# In[1]:


project_name = 'wgan_face_generation'


# ## WGAN modified of DCGAN in:
# 1. remove sigmoid in the last layer of discriminator(classification -> regression)                                    
# 2. no log Loss (Wasserstein distance)
# 
# 3. clip param norm to c (Wasserstein distance and Lipschitz continuity)
# 
# 4. No momentum-based optimizer, use RMSProp，SGD instead

# In[2]:


import numpy as np
import pandas as pd
import os

import torch.utils.data
from PIL import Image
from typing import Any, Tuple

from torch.utils.data import DataLoader


# In[3]:


def compress_image(prev_image, n):
    if n < 2:
        return prev_image

    height = prev_image.shape[0] // n
    width = prev_image.shape[1] // n
    new_image = np.zeros((height, width), dtype="uint8")
    for i in range(0, height):
        for j in range(0, width):
            new_image[i, j] = prev_image[n * i, n * j]

    return new_image

class CEMDataset(torch.utils.data.Dataset):
    DATASETS_TRAIN = [
        'binary_501',
        'binary_502',
#         'binary_503',
#         'binary_504',
#         'binary_505',
#         'binary_506',
#         'binary_507',
#         'binary_508',
#         'binary_509',
#         'binary_510',
#         'binary_511',
#         'binary_512',
#         'binary_1001',
#         'binary_1002',
#         'binary_1003',
#         'binary_rl_fix_501',
#         'binary_rl_fix_502',
#         'binary_rl_fix_503',
#         'binary_rl_fix_504',
#         'binary_rl_fix_505',
#         'binary_rl_fix_506',
#         'binary_rl_fix_507',
#         'binary_rl_fix_508',
#         'binary_rl_fix_509',
#         'binary_rl_fix_510',
#         'binary_rl_fix_511',
#         'binary_rl_fix_512',
#         'binary_rl_fix_513',
#         'binary_rl_fix_514',
#         'binary_rl_fix_515',
#         'binary_rl_fix_516',
#         'binary_rl_fix_517',
#         'binary_rl_fix_518',
#         'binary_rl_fix_519',
#         'binary_rl_fix_520',
#         'binary_rl_fix_1001',
#         'binary_rl_fix_1002',
#         'binary_rl_fix_1003',
#         'binary_rl_fix_1004',
#         'binary_rl_fix_1005',
#         'binary_rl_fix_1006',
#         'binary_rl_fix_1007',
#         'binary_rl_fix_1008',
    ]

    DATASETS_VALID = [
        'binary_1004',
        'binary_test_1001',
        'binary_test_1002',
        'binary_rl_fix_1009',
        'binary_rl_fix_1010',
        'binary_rl_fix_1011',
        'binary_rl_fix_1012',
        'binary_rl_fix_1013',
        'binary_rl_fix_test_1001',
    ]

    DATASETS_TEST = [
        'binary_new_test_501',
#         'binary_new_test_1501',
#         'binary_rl_fix_1014',
#         'binary_rl_fix_1015',
#         'binary_rl_fix_test_1002',
#         'binary_rl_fix_test_1003',
#         'binary_rl_fix_test_1004',
#         'binary_rl_fix_test_1005',
#         'binary_test_1101',
    ]

    def __init__(self,
                 root: str,
                train: bool = True,
                scale: int = 1,
                 ) -> None:
        self.train = train
        self.root = root
        self.scale = scale
        self.width = 200 // scale
        self.height = 100 // scale

        if self.train:
            DATAPATH = os.path.join(root, 'train')
            DATASETS = self.DATASETS_TRAIN
        else:
            DATAPATH = os.path.join(root, 'test')
            DATASETS = self.DATASETS_TEST

        self.data: Any = []
        self.targets = []

        print('data loading ... ')

        # load Train dataset
        for data in DATASETS:
            dataframe = pd.read_csv(os.path.join(DATAPATH, '{}.csv'.format(data)), delim_whitespace=False, header=None)
            dataset = dataframe.values

            # split into input (X) and output (Y) variables
            fileNames = dataset[:, 0]

            # 1. first try max
            dataset[:, 1:25] /= 2767.1
            self.targets.extend(dataset[:, 1:25])
            for idx, file in enumerate(fileNames):
                try:
                    image = Image.open(os.path.join(DATAPATH, data, '{}.tiff'.format(int(file))))
                    image = np.array(image).astype(np.uint8)
                except (TypeError, FileNotFoundError) as te:
                    image = Image.open(os.path.join(DATAPATH, data, '{}.tiff'.format(idx + 1)))
                    try:
                        image = np.array(image).astype(np.uint8)
                    except:
                        continue
                image = compress_image(image, self.scale)
                self.data.append(np.array(image).flatten(order='C'))

        self.data = np.vstack(self.data).reshape(-1, 1, self.height, self.width)
        self.data = self.data.transpose((0, 1, 2, 3))  # convert to HWC CHW
        print(f'Data Loading Finished. len : {len(self.data)}')


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target

    def __len__(self) -> int:
        return len(self.data)


# In[4]:


data_dir = os.path.join(os.getcwd(), 'maxwellfdfd')
data_dir


# In[5]:


# import os

# DATA_DIR = '../input/anime-faces'
# print(os.listdir(DATA_DIR))


##
# Data
# data_dir = '../maxwellfdfd'

cem_train = CEMDataset(data_dir, train=True, scale=5)
# cem_unlabeled = CEMDataset('./maxwellfdfd', train=True, scale=5)
cem_test = CEMDataset(data_dir, train=False, scale=5)

dataset_size = {'train': len(cem_train), 'test': len(cem_test)}


# In[6]:


checkpoint_dir = os.path.join(os.getcwd(), 'cem', 'train', 'weights')
# checkpoint_dir = os.path.join('./cem', 'train', 'weights')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# In[7]:


# print(os.listdir(DATA_DIR+'/data')[:10])


# Let's load this dataset using the `ImageFolder` class from `torchvision`. We will also resize and crop the images to 64x64 px, and normalize the pixel values with a mean & standard deviation of 0.5 for each channel. This will ensure that pixel values are in the range `(-1, 1)`, which is more  convenient for training the discriminator. We will also create a data loader to load the data in batches.

# In[8]:


from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T


# In[9]:


image_size = 64
batch_size = 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

#stats = (1,1,1), (1, 1, 1)


# In[10]:


train_dl = DataLoader(cem_train, batch_size, shuffle=True, pin_memory=True)


# Let's create helper functions to denormalize the image tensors and display some sample images from a training batch.

# In[11]:


import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


def denorm(img_tensors):
    return img_tensors * 1.


# In[13]:


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    s = make_grid(images.detach()[:nmax], nrow=8)
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8, padding=5, pad_value=0.5).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break


# In[14]:


show_batch(train_dl)


# ## Using a GPU
# 
# To seamlessly use a GPU, if one is available, we define a couple of helper functions (`get_default_device` & `to_device`) and a helper class `DeviceDataLoader` to move our model & data to the GPU, if one is available.

# In[15]:


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# Based on where you're running this notebook, your default device could be a CPU (`torch.device('cpu')`) or a GPU (`torch.device('cuda')`).

# In[16]:


device = get_default_device()
device


# We can now move our training data loader using `DeviceDataLoader` for automatically transferring batches of data to the GPU (if available).

# In[17]:


train_dl = DeviceDataLoader(train_dl, device)


# ## Discriminator Network
# 
# The discriminator takes an image as input, and tries to classify it as "real" or "generated". In this sense, it's like any other neural network. We'll use a convolutional neural networks (CNN) which outputs a single number output for every image. We'll use stride of 2 to progressively reduce the size of the output feature map.
# 
# ![](https://github.com/vdumoulin/conv_arithmetic/raw/master/gif/padding_strides_odd.gif)

# In[19]:


import torch.nn as nn


# In[226]:


discriminator = nn.Sequential(
    # in: 1 x 20 x 40

    nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 64 x 32 x 32

    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 128 x 16 x 16

    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 256 x 8 x 8

    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 512 x 4 x 4

    nn.Conv2d(512, 24, kernel_size=4, stride=1, padding=0, bias=False),
    # out: 1 x 1 x 1
    
    # Modification 1: remove sigmoid
    # nn.Sigmoid()

    )
print(discriminator)


# Note that we're using the Leaky ReLU activation for the discriminator.
# 
# <img src="https://cdn-images-1.medium.com/max/1600/1*ypsvQH7kvtI2BhzR2eT_Sw.png" width="420">
# 
# 
# >  Different from the regular ReLU function, Leaky ReLU allows the pass of a small gradient signal for negative values. As a result, it makes the gradients from the discriminator flows stronger into the generator. Instead of passing a gradient (slope) of 0 in the back-prop pass, it passes a small negative gradient.  - [Source](https://sthalles.github.io/advanced_gans/)
# 
# Just like any other binary classification model, the output of the discriminator is a single number between 0 and 1, which can be interpreted as the probability of the input image being real i.e. picked from the original dataset.
# 
# Let's move the discriminator model to the chosen device.

# In[21]:


discriminator = to_device(discriminator, device)


# ## Generator Network
# 
# The input to the generator is typically a vector or a matrix of random numbers (referred to as a latent tensor) which is used as a seed for generating an image. The generator will convert a latent tensor of shape `(128, 1, 1)` into an image tensor of shape `3 x 28 x 28`. To achive this, we'll use the `ConvTranspose2d` layer from PyTorch, which is performs to as a *transposed convolution* (also referred to as a *deconvolution*). [Learn more](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md#transposed-convolution-animations)
# 
# ![](https://i.imgur.com/DRvK546.gif)

# In[87]:


latent_size = 128


# In[204]:


generator = nn.Sequential(
    # in: latent_size x 1 x 1

    nn.ConvTranspose2d(latent_size, 256, kernel_size=(3,5), stride=1, padding=0, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    # out: 512 x 4 x 4

    #nn.ConvTranspose2d(512, 256, kernel_size=(3,4), stride=2, padding=1, bias=False),
    #nn.BatchNorm2d(256),
    #nn.ReLU(True),
    # out: 256 x 8 x 8

    nn.ConvTranspose2d(256, 128, kernel_size=(3,4), stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # out: 128 x 16 x 16

    nn.ConvTranspose2d(128, 64, kernel_size=(3,4), stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    # out: 64 x 32 x 32

    nn.ConvTranspose2d(64, 1, kernel_size=(4,4), stride=2, padding=(0, 1), bias=False),
    nn.Tanh()
    # out: 1 x 20 x 40
)


# We use the TanH activation function for the output layer of the generator.
# 
# <img src="https://nic.schraudolph.org/teach/NNcourse/figs/tanh.gif" width="420" >
# 
# > "The ReLU activation (Nair & Hinton, 2010) is used in the generator with the exception of the output layer which uses the Tanh function. We observed that using a bounded activation allowed the model to learn more quickly to saturate and cover the color space of the training distribution. Within the discriminator we found the leaky rectified activation (Maas et al., 2013) (Xu et al., 2015) to work well, especially for higher resolution modeling." - [Source](https://stackoverflow.com/questions/41489907/generative-adversarial-networks-tanh)
# 
# 
# Note that since the outputs of the TanH activation lie in the range `[-1,1]`, we have applied the similar transformation to the images in the training dataset. Let's generate some outputs using the generator and view them as images by transforming and denormalizing the output.

# In[92]:


def weight_init(m):
    # weight_initialization: important for wgan
    class_name=m.__class__.__name__
    if class_name.find('Conv')!=-1:
        m.weight.data.normal_(0,0.02)
    elif class_name.find('Norm')!=-1:
        m.weight.data.normal_(1.0,0.02)


# In[224]:


discriminator.apply(weight_init)
generator.apply(weight_init)


# In[207]:


xb = torch.randn(batch_size, latent_size, 1, 1) # random latent tensors
fake_images = generator(xb)
print(fake_images.shape)
show_images(fake_images)


# As one might expect, the output from the generator is basically random noise, since we haven't trained it yet. 
# 
# Let's move the generator to the chosen device.

# In[208]:


generator = to_device(generator, device)


# ## Discriminator Training
# 
# 

# In[222]:


def train_discriminator(real_images, opt_d):
    # Clear discriminator gradients
    opt_d.zero_grad()

    # Pass real images through discriminator
    real_preds = discriminator(real_images.float())
    
    print('real_preds', real_preds)
    #modification: remove binary cross entropy
        #real_targets = torch.ones(real_images.size(0), 1, device=device)
        #real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_loss = -torch.mean(real_preds)
        
    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Pass fake images through discriminator
    #modification: remove binary cross entropy
        #fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = discriminator(fake_images)
        #fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_loss = torch.mean(fake_preds)

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_loss.item(), fake_loss.item()


# Here are the steps involved in training the discriminator.
# 
# - We expect the discriminator to output 1 if the image was picked from the real MNIST dataset, and 0 if it was generated using the generator network. 
# 
# - We first pass a batch of real images, and compute the loss, setting the target labels to 1. 
# 
# - Then we pass a batch of fake images (generated using the generator) pass them into the discriminator, and compute the loss, setting the target labels to 0. 
# 
# - Finally we add the two losses and use the overall loss to perform gradient descent to adjust the weights of the discriminator.
# 
# It's important to note that we don't change the weights of the generator model while training the discriminator (`opt_d` only affects the `discriminator.parameters()`)

# ## Generator Training
# 
# Since the outputs of the generator are images, it's not obvious how we can train the generator. This is where we employ a rather elegant trick, which is to use the discriminator as a part of the loss function. Here's how it works:
# 
# - We generate a batch of images using the generator, pass the into the discriminator.
# 
# - We calculate the loss by setting the target labels to 1 i.e. real. We do this because the generator's objective is to "fool" the discriminator. 
# 
# - We use the loss to perform gradient descent i.e. change the weights of the generator, so it gets better at generating real-like images to "fool" the discriminator.
# 
# Here's what this looks like in code.

# In[210]:


def train_generator(opt_g):
    # Clear generator gradients
    opt_g.zero_grad()
    
    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)
    
    # Try to fool the discriminator
    preds = discriminator(fake_images)
    #modificationL remove binary cross entropy
        #targets = torch.ones(batch_size, 1, device=device)
    loss = -torch.mean(preds)
    
    # Update generator weights
    loss.backward()
    opt_g.step()
    
    return loss.item()


# Let's create a directory where we can save intermediate outputs from the generator to visually inspect the progress of the model. We'll also create a helper function to export the generated images.

# In[211]:


from torchvision.utils import save_image


# In[212]:


sample_dir = 'generated'
os.makedirs(sample_dir, exist_ok=True)


# In[213]:


def save_samples(index, latent_tensors, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))


# We'll use a fixed set of input vectors to the generator to see how the individual generated images evolve over time as we train the model. Let's save one set of images before we start training our model.

# In[214]:


fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)


# In[215]:


save_samples(0, fixed_latent)


# ## Full Training Loop
# 
# Let's define a `fit` function to train the discriminator and generator in tandem for each batch of training data. We'll use the Adam optimizer with some custom parameters (betas) that are known to work well for GANs. We will also save some sample generated images at regular intervals for inspection.
# 
# 
# 

# In[216]:


from tqdm.notebook import tqdm
import torch.nn.functional as F


# In[217]:


def fit(epochs, lr, start_idx=1):
    torch.cuda.empty_cache()
    
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    # Create optimizers
    opt_d = torch.optim.RMSprop(discriminator.parameters(), lr=lr)
    opt_g = torch.optim.RMSprop(generator.parameters(), lr=lr)
    
    
    
    for epoch in range(epochs):
        for real_images, _ in tqdm(train_dl):
            # Train discriminator
            # modification: clip param for discriminator
            for parm in discriminator.parameters():
                parm.data.clamp_(-clamp_num, clamp_num)
            loss_d, real_score, fake_score = train_discriminator(real_images, opt_d)
            # Train generator
            loss_g = train_generator(opt_g)
            
        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)
        
        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
    
        # Save generated images
        save_samples(epoch+start_idx, fixed_latent, show=False)
    
    return losses_g, losses_d, real_scores, fake_scores


# We are now ready to train the model. Try different learning rates to see if you can maintain the fine balance between the training the generator and the discriminator.

# In[218]:


lr = 0.00005
epochs = 200
clamp_num=0.01# WGAN clip gradient


# In[223]:


history = fit(epochs, lr)


# In[ ]:


losses_g, losses_d, real_scores, fake_scores = history


# Now that we have trained the models, we can save checkpoints.

# In[ ]:


# Save the model checkpoints 
torch.save(generator.state_dict(), 'G.pth')
torch.save(discriminator.state_dict(), 'D.pth')


# Here's how the generated images look, after the 1st, 5th and 10th epochs of training.

# In[ ]:


from IPython.display import Image


# In[ ]:


Image('./generated/generated-images-0001.png')


# We can visualize the training process by combining the sample images generated after each epoch into a video using OpenCV.

# In[ ]:


import cv2
import os

vid_fname = 'gans_training.avi'

files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if 'generated' in f]
files.sort()

out = cv2.VideoWriter(vid_fname,cv2.VideoWriter_fourcc(*'MP4V'), 1, (530,530))
[out.write(cv2.imread(fname)) for fname in files]
out.release()


# Here's what it looks like:
# 
# ![]()
# 
# 
# We can also visualize how the loss changes over time. Visualizing 
# losses is quite useful for debugging the training process. For GANs, we expect the generator's loss to reduce over time, without the discriminator's loss getting too high.
# 
# 

# In[ ]:


plt.plot(losses_d, '-')
plt.plot(losses_g, '-')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Discriminator', 'Generator'])
plt.title('Losses');


# In[ ]:


plt.plot(real_scores, '-')
plt.plot(fake_scores, '-')
plt.xlabel('epoch')
plt.ylabel('score')
plt.legend(['Real', 'Fake'])
plt.title('Scores');

