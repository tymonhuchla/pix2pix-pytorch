import torch as T
import matplotlib.pyplot as plt


def save_gen(state, path, path_for_best=None):
    T.save(state, path)
    if path_for_best:
        T.save(state, path_for_best)


def save_disc(state, path, path_for_best=None):
    T.save(state, path)
    if path_for_best:
        T.save(state, path_for_best)


def show_trio(img0, img1, img2, idx=None, epoch=None, show=False):
    fig, ax = plt.subplots(1, 3, figsize=(12,12))
    img0, img1, img2 = img0* 0.5 + 0.5, img1* 0.5 + 0.5, img2* 0.5 + 0.5
    ax[0].imshow(img0.permute(1,2,0)[:,:,0], cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Input Image')
    ax[1].imshow(img1.permute(1,2,0))
    ax[1].axis('off')
    ax[1].set_title('Predicted Image')
    ax[2].imshow(img2.permute(1,2,0))
    ax[2].axis('off')
    ax[2].set_title('Ground truth')
    if idx:
        plt.savefig(f'/content/drive/MyDrive/pix2pix/imgs/fig_{epoch}_{idx}.png')
    if show:
        plt.show()


def show_pair(img0, img1):
    fig, ax = plt.subplots(1, 2, figsize=(12,12))
    ax[0].imshow(img0.permute(1,2,0)[:,:,0] * 0.5 + 0.5, cmap='gray')
    ax[0].axis('off')
    ax[1].imshow(img1.permute(1,2,0) * 0.5 + 0.5)
    ax[1].axis('off')
    plt.show()