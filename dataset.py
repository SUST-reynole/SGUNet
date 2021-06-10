import os
import PIL.Image as Image
import torch.utils.data as data



def make_dataset(raw,masks):
    imgs=[]
    n = len(os.listdir(raw))
    ls = os.listdir(raw)
    for i in range(n):
        img=os.path.join(raw,ls[i])
        mask=os.path.join(masks,ls[i])
        imgs.append((img,mask))
    return imgs


class Datasets(data.Dataset):
    def __init__(self, raw,mask, transform=None, target_transform=None):
        imgs = make_dataset(raw,mask)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_x = img_x.resize((256,256))
        img_x = img_x.convert('L')
        img_y = Image.open(y_path)
        img_y = img_y.resize((256,256))
        img_y = img_y.convert('L')
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)
