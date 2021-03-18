import torch as T
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, features):
        super(Discriminator, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(1 + 3, features[0], kernel_size=(4, 4), stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.LeakyReLU(0.2)
        )

        self.net = nn.Sequential(
            self.block(features[0], features[1]),
            self.block(features[1], features[2]),
            self.block(features[2], features[3], stride=1),
            nn.Conv2d(features[-1], 1, kernel_size=(4, 4), stride=1, padding=1, padding_mode='reflect')
        )

    @staticmethod
    def block(in_channels, out_channels, kernel_size=(4,4), stride=2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x0, x1):
        x0 = T.cat([x0, x1], dim=1)
        x0 = self.input_layer(x0)
        x0 = self.net(x0)
        return x0


class Encoder(nn.Module):
    def __init__(self, out_channels):
        super(Encoder, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(1, out_channels, 4, 2, 1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )

        self.down1 = self.block(out_channels, out_channels * 2)
        self.down2 = self.block(out_channels * 2, out_channels * 4)
        self.down3 = self.block(out_channels * 4, out_channels * 8)
        self.down_repeat = self.block(out_channels * 8, out_channels * 8)

    @staticmethod
    def block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        down0 = self.input_layer(x)
        down1 = self.down1(down0)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down_repeat(down3)
        down5 = self.down_repeat(down4)
        down6 = self.down_repeat(down5)
        return down6, [down0, down1, down2, down3, down4, down5, down6]


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 8, out_channels * 8, 4, 2, 1, padding_mode='reflect'),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up0 = self.block(out_channels * 8, out_channels * 8, dropout=True)
        self.up_repeat = self.block(out_channels * 8 * 2, out_channels * 8, dropout=True)
        self.up1 = self.block(out_channels * 8 * 2 , out_channels * 4, dropout=False)
        self.up2 = self.block(out_channels * 4 * 2, out_channels * 2, dropout=False)
        self.up3 = self.block(out_channels * 2 * 2, out_channels, dropout=False)
        self.output = nn.Sequential(
            nn.ConvTranspose2d(out_channels * 2, in_channels, 4, 2, 1),
            nn.Tanh()
        )

    @staticmethod
    def block(in_channels, out_channels, dropout=False):
        dropout = nn.Dropout(0.5) if dropout else nn.Identity()
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            dropout,
        )

    def forward(self, x, enc_layers):
        up0 = self.up0(x)
        up1 = self.up_repeat(T.cat((up0, enc_layers[-1]), dim=1))
        up2 = self.up_repeat(T.cat((up1, enc_layers[-2]), dim=1))
        up3 = self.up_repeat(T.cat((up2, enc_layers[-3]), dim=1))
        up4 = self.up1(T.cat((up3, enc_layers[-4]), dim=1))
        up5 = self.up2(T.cat((up4, enc_layers[-5]), dim=1))
        up6 = self.up3(T.cat((up5, enc_layers[-6]), dim=1))
        return self.output(T.cat((up6, enc_layers[-7]), dim=1))


class Generator(nn.Module):
    def __init__(self, encoder, decoder, bottleneck, in_channels, out_channels):
        super(Generator, self).__init__()
        self.encoder = encoder(out_channels)
        self.decoder = decoder(in_channels, out_channels)
        self.bottleneck = bottleneck(out_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x, layers = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, layers)
        return x