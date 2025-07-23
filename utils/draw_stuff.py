import matplotlib.pyplot as plt
import numpy as np

def draw_img(data, type='depth'):

    if type == 'fluorescence' or type == 'reflectance':
        fig, axs = plt.subplots(1, 6, figsize=(10, 5))
        for i in range(6):
            axs[i].imshow(data[:, :, i], cmap='hot', interpolation='nearest')
            axs[i].axis('off')
        plt.show()
    elif type == 'optical_props':
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(data[:, :, 0], cmap='hot', interpolation='nearest')
        axs[0].set_title('mu_a')
        axs[0].axis('off')
        axs[1].imshow(data[:, :, 1], cmap='hot', interpolation='nearest')
        axs[1].set_title('mu_sp')
        axs[1].axis('off')
        plt.show()
    elif type == 'depth':
        plt.imshow(data, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()
    else:
        plt.imshow(data, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()

def draw_hist(data, type):
    if type == 'fluorescence':
        fig, axs = plt.subplots(1, 6, figsize=(10,5))
        for i in range(6):
            axs[i].hist(data[:, :, i].ravel(), bins=100, color='green', edgecolor='black')
            axs[i].set_title(f'Channel {i+1}')
        plt.show()
    elif type == 'depth':
        plt.hist(data.ravel(), bins=100, color='green', edgecolor='black')
        plt.show()

def get_stats(data):
    draw_img(data['fluorescence'], type='fluorescence')
    draw_img(data['op'], type='optical_props')
    draw_img(data['depth'], type='depth')

    draw_hist(data['fluorescence'], type='fluorescence')
    draw_hist(data['depth'], type='depth')