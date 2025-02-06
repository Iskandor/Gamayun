import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from grokfast_pytorch import GrokFastAdamW
from torch import optim
from tqdm import tqdm

from loss.SNDv3Loss import SNDv3Loss


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc(x)
        return x


class TestPipeline:
    def __init__(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        batch_size = 4096

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

        self.encoder = Encoder()
        self.classifier = Classifier()
        self.projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.device = 'cuda:2'


    def evaluate_model(self, dataloader):
        model = nn.Sequential(
            self.encoder,
            self.classifier
        )
        model = model.to(self.device)
        model.eval()

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = model(images.to(self.device))
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(self.device)).sum().item()

        return correct, total

    def train_model(self, epochs):
        model = nn.Sequential(
            self.encoder,
            self.classifier
        )
        model = model.to(self.device)
        model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=5e-4)

        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = []
            for i, data in tqdm(enumerate(self.trainloader, 0), desc='Epoch {0:d}'.format(epoch), total=len(self.trainloader)):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs.to(self.device))
                loss = criterion(outputs, labels.to(self.device))
                loss.backward()
                optimizer.step()
                # scheduler.step()

                # print statistics
                running_loss.append(loss.detach())
            print('Loss: {0:.3f}, LR:{1:.3f}'.format(torch.stack(running_loss).mean(), optimizer.param_groups[0]['lr']))

    def pretrain_model(self, epochs):
        model = nn.Sequential(
            self.encoder,
            self.projection
        )
        model = model.to(self.device)
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=5e-4)
        criterion = SNDv3Loss(None, model)

        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = []
            for i, data in tqdm(enumerate(self.trainloader, 0), desc='Epoch {0:d}'.format(epoch), total=len(self.trainloader)):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs.to(self.device))
                # sim_loss, vc_loss = criterion.metric_loss(inputs.to(self.device), outputs)
                # loss = sim_loss + vc_loss
                # loss = criterion.topology_loss(inputs.to(self.device), outputs)
                loss = criterion.ssim_loss(inputs.to(self.device), outputs)
                loss.backward()
                optimizer.step()
                # scheduler.step()

                # print statistics
                running_loss.append(loss.detach())
            print('Loss: {0:.3f}, LR:{1:.3f}'.format(torch.stack(running_loss).mean(), optimizer.param_groups[0]['lr']))

    def finetune_model(self, epochs):
        model = nn.Sequential(
            self.encoder,
            self.classifier
        )
        model = model.to(self.device)
        model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.classifier.parameters(), lr=3e-3, weight_decay=5e-4)

        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = []
            for i, data in tqdm(enumerate(self.trainloader, 0), desc='Epoch {0:d}'.format(epoch), total=len(self.trainloader)):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.classifier(self.encoder(inputs.to(self.device)).detach())
                loss = criterion(outputs, labels.to(self.device))
                loss.backward()
                optimizer.step()
                # scheduler.step()

                # print statistics
                running_loss.append(loss.detach())
            print('Loss: {0:.3f}, LR:{1:.3f}'.format(torch.stack(running_loss).mean(), optimizer.param_groups[0]['lr']))


if __name__ == '__main__':
    # optimizer = GrokFastAdamW(
    #     model.parameters(),
    #     lr=3e-3,
    #     weight_decay=5e-4
    # )

    # scheduler = optim.lr_scheduler.CyclicLR(optimizer)

    pipeline = TestPipeline()
    # pipeline.train_model(100)
    pipeline.pretrain_model(100)
    torch.save(pipeline.encoder.state_dict(), './cifar10_encoder.pt')
    pipeline.finetune_model(10)

    correct, total = pipeline.evaluate_model(pipeline.trainloader)
    print(f'Accuracy of the network on the train images: {100 * correct // total} %')
    correct, total = pipeline.evaluate_model(pipeline.testloader)
    print(f'Accuracy of the network on the test images: {100 * correct // total} %')
    print('Finished Training')
