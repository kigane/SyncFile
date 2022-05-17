from PIL import Image, ImageOps
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
import cv2 as cv

def rgb2lab_show(path):
    im = Image.open(path).convert('RGB')
    # im = ImageOps.grayscale(im)
    im = np.array(im)
    # im = np.tile(im, (3, 1, 1)).transpose(1, 2, 0)
    lab = color.rgb2lab(im).astype(np.float32)
    L = lab[..., 0]
    A = lab[..., 1]
    B = lab[..., 2]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    axes = axes.reshape(-1)
    for ax, img in zip(axes, [L, A, B]):
        ax.hist(img.reshape(-1), bins=256)
        # ax.imshow(img, cmap='gray')
        # ax.set_axis_off()
    # axes[3].hist(L.reshape(-1), bins=256)
    plt.show()

if __name__ == '__main__':
    rgb2lab_show('10001570.jpg')
    # rgb2lab_show('20220428181759721.png')
