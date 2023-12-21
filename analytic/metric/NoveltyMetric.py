import os
import random
import zipfile
from glob import glob
from pathlib import Path

import PIL
import requests
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class NoveltyMetricDataset(Dataset):
    def __init__(self, root) -> None:
        self.data = []
        transform = transforms.Compose([transforms.PILToTensor()])

        for file in glob(str(root) + '/*.png'):
            image = transform(PIL.Image.open(file)).float()
            image /= 255.
            self.data.append(image)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class NoveltyMetric:
    KEY = 'NoveltyMetric:V2'

    def __init__(self, width, height, predictor, target, batch_size, device):
        self.width = width
        self.height = height
        self.predictor = predictor
        self.target = target
        self.device = device
        self.root = Path(os.getcwd(), 'analytic', 'metric', 'dataset')

        self.data = []
        self.deploy_images()

        self.dataloader = self.init_dataloader(batch_size)

    def draw_shapes(self):
        factor = self.width // 4

        for i in range(factor):
            for m in range(2, factor):
                self.data.append(self.draw(i, modulo=m))

    def draw(self, i, modulo=2):
        shapes = ['rectangle', 'ellipse']
        out = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(out)
        for j in range((i + 1) * (i + 1)):
            # scale_x = int(self.width // (i + 1))
            scale_x = int(self.width // (i + 1) * random.uniform(0., 3.))
            # scale_y = int(self.height // (i + 1))
            scale_y = int(self.height // (i + 1) * random.uniform(0., 3.))
            color = random.randint(63, 255)
            x = j % (i + 1)
            y = j // (i + 1)
            if x % modulo == y % modulo:
                shape = shapes[random.randint(0, len(shapes) - 1)]
                if shape == 'rectangle':
                    draw.rectangle([x * scale_x, y * scale_y, (x + 1) * scale_x, (y + 1) * scale_y], fill=color)
                if shape == 'ellipse':
                    draw.ellipse([x * scale_x, y * scale_y, (x + 1) * scale_x, (y + 1) * scale_y], fill=color)
        # out.show()
        return out

    def invert_images(self):
        inverted_data = []
        for im in self.data:
            inverted_data.append(ImageOps.invert(im))

        self.data += inverted_data

    def deploy_images(self):
        if len(os.listdir(self.root)) == 0:
            self.draw_shapes()
            self.invert_images()
            print('Deploying dataset')
            for i, img in enumerate(self.data):
                filename = self.root / Path('img_{0}.png'.format(i))
                print(filename)
                img.save(filename)
        if len(os.listdir(self.root)) == 1:
            for file in glob(self.root.__str__() + '/*.zip'):
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    zip_ref.extractall(self.root)

    def init_dataloader(self, batch_size):
        dataset = NoveltyMetricDataset(self.root.__str__())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def value(self):
        error = []

        for sample in self.dataloader:
            with torch.no_grad():
                zp, zt = self.predictor(sample.to(self.device)), self.target(sample.to(self.device))

            error.append(F.mse_loss(zp, zt))

        error = torch.stack(error).mean()

        return error


if __name__ == "__main__":
    metric = NoveltyMetric(96, 96, None, None, 512, 'cpu')
    print(metric)
