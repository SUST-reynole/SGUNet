import torch.nn as nn
import torchvision.transforms as transforms
import PIL.Image as Image
import torch

class conv_encoder(nn.Module):
    def __init__(self):
        super(conv_encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # (b, 16, 10, 10)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # (b, 16, 5, 5)
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # (b, 8, 3, 3)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # (b, 8, 2, 2)
        )


    def forward(self, x):
        encode = self.encoder(x)
        return encode

class conv_decoder(nn.Module):
    def __init__(self):
        super(conv_decoder, self).__init__()


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # (b, 16, 5, 5)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # (b, 8, 15, 15)
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 2, 2, stride=2, padding=1),  # (b, 1, 28, 28)
            #nn.Sigmoid()
        )

    def forward(self, x):
        decode = self.decoder(x)
        return decode

class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # (b, 16, 10, 10)
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d(2, stride=2),  # (b, 16, 5, 5)
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # (b, 8, 3, 3)
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d(2, stride=1)  # (b, 8, 2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # (b, 16, 5, 5)
            nn.LeakyReLU(0.2,True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # (b, 8, 15, 15)
            nn.LeakyReLU(0.2,True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # (b, 1, 28, 28)
            nn.LeakyReLU(0.2,True)
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode


if __name__=="__main__":
    print("coding happy!")
    # root_train = ""
    # img = Image.open(root_train)
    # img = img.resize((256,256))
    # y_transforms = transforms.ToTensor()
    # img = y_transforms(img)
    # img = img.unsqueeze(0)
    img = torch.randn(1, 1, 256, 256)
    encoder = conv_encoder()
    decoder = conv_decoder()
    vec = encoder(img)
    new_img = decoder(vec)
    print()
