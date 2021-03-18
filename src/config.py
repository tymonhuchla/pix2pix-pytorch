import torch as T

SHAPE = 256
BATCH_SIZE = 16
NUM_WORKERS = 2
rootdir = '/content/data/'
DEVICE = 'cuda' if T.cuda.is_available() else 'cpu'
LR = 2e-4
BETA1 = 0.5
BETA2 = 0.99
LAMBDA_L1 = 100
EPOCHS = 500
PATH = '/content/drive/MyDrive/celebA/img_align_celeba.zip'
