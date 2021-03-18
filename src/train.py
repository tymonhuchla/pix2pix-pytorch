import torch.nn as nn
from config import *
from dataloader import loader
from models import Generator, Discriminator, Encoder, Decoder, Bottleneck
from utils import *

features = [64,128,256,512]
discriminator = Discriminator(features).to(DEVICE)
generator = Generator(Encoder, Decoder, Bottleneck, 3, 64).to(DEVICE)

gen_optimizer = T.optim.Adam(generator.parameters(), LR, betas=(BETA1, BETA2))
disc_optimizer = T.optim.Adam(discriminator.parameters(), LR, betas=(BETA1, BETA2))

bce_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()

g_scaler = T.cuda.amp.GradScaler()
d_scaler = T.cuda.amp.GradScaler()

g_losses = []
d_losses = []

for epoch in range(1, EPOCHS+1):

    print(f'EPOCH [{epoch}/{EPOCHS}]')

    for idx, (x0, x1) in enumerate(loader):

        x0, x1 = x0[0].to(DEVICE), x1[0].to(DEVICE)

        with T.cuda.amp.autocast():

            fake = generator(x0)
            D_real = discriminator(x0, x1)
            D_fake = discriminator(x0, fake)

            D_real_loss = bce_loss(D_real, T.ones_like(D_real))
            D_fake_loss = bce_loss(D_fake, T.zeros_like(D_fake))
            D_loss = (D_fake_loss + D_real_loss) / 2

        discriminator.zero_grad()
        d_scaler.scale(D_loss).backward(retain_graph=True)
        d_scaler.step(disc_optimizer)
        d_scaler.update()

        with T.cuda.amp.autocast():

            D_fake = discriminator(x0, fake)
            G_fake_loss = bce_loss(D_fake, T.ones_like(D_fake))
            L1 = l1_loss(fake, x1) * LAMBDA_L1
            G_loss = L1 + G_fake_loss

        generator.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(gen_optimizer)
        g_scaler.update()

        if idx % 100 == 0:
            print(f'Batch [{idx}/{loader.__len__()}]')
            print('Discriminator loss {}'.format(D_loss))
            print('Generator loss {}'.format(G_loss))
        with T.no_grad():
            to_visualize = generator(x0).detach().cpu()
        input_img = x0[0].detach().cpu()
        ground_truth = x1[0].detach().cpu()
        show_trio(input_img, to_visualize[0], ground_truth, idx, epoch)

        BEST_PATH_DISC = f'/content/drive/MyDrive/pix2pix/weights/Bdisc_{epoch}_{idx}.pth'
        REGULAR_PATH_DISC = f'/content/drive/MyDrive/pix2pix/weights/disc_{epoch}_{idx}.pth'

        BEST_PATH_GEN = f'/content/drive/MyDrive/pix2pix/weights/Bgen_{epoch}_{idx}.pth'
        REGULAR_PATH_GEN = f'/content/drive/MyDrive/pix2pix/weights/gen_{epoch}_{idx}.pth'

        g_losses.append(G_loss.detach().cpu())
        d_losses.append(D_loss.detach().cpu())

        if idx > 1:
            if D_loss.detach().cpu() == min(d_losses):
                save_disc(discriminator.state_dict(), REGULAR_PATH_DISC, BEST_PATH_DISC)
                print('New best discriminator')

            if G_loss.detach().cpu() == min(g_losses):
                save_gen(generator.state_dict(), REGULAR_PATH_GEN, BEST_PATH_GEN)
                print('New best generator')