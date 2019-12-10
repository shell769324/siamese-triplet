
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.nn as nn

import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from trainer import fit
import numpy as np
cuda = torch.cuda.is_available()

import matplotlib
import matplotlib.pyplot as plt
from networks import EmbeddingNet, ClassificationNet
from metrics import AccumulatedAccuracyMetric
from datasets import TripletFolder
batch_size = 32
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
train_dataset = torchvision.datasets.ImageFolder(
    root = './FID-prepped/train',
    transform=transforms.Compose(
        [
         
        torchvision.transforms.Resize((224,400)),
        torchvision.transforms.RandomRotation(3),
         torchvision.transforms.ToTensor(),
         transforms.Lambda(lambda x: x/255.),   
         normalize,
        ]
    )
)
test_dataset = torchvision.datasets.ImageFolder(
    root = './FID-prepped/test',
    transform=transforms.Compose(
        [
        torchvision.transforms.Resize((224,400)),
         torchvision.transforms.ToTensor(),
         transforms.Lambda(lambda x: x/255.),   
         normalize,
        ]
    )
)
# LAST THING TO DO: MAKE THE REF DATASET
ref_dataset = torchvision.datasets.ImageFolder(
    root = './FID-prepped/ref',
    transform=transforms.Compose(
        [
        torchvision.transforms.Resize((224,400)),
         torchvision.transforms.ToTensor(),
         transforms.Lambda(lambda x: x/255.),   
         normalize,
        ]
    )
)

triplet_train_dataset = TripletFolder(train_dataset, True)
triplet_test_dataset = TripletFolder(test_dataset, False)
triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
ref_loader = torch.utils.data.DataLoader(ref_dataset, batch_size=batch_size, shuffle=False, **kwargs)
# Set up the network and training parameters
from networks import EmbeddingNet, TripletNet
from losses import TripletLoss

margin = 100.
#embedding_net = EmbeddingNet()
import torchvision.models as models
# THE MODEL
resnet = models.resnet18(pretrained = True)
#num_ftrs = resnet.fc.in_features
#resnet.fc = nn.Linear(num_ftrs, 3)
count = 0
'''
for child in resnet.children():
    count += 1
i = 0
for child in resnet.children():
    if i < count - 1:
        for param in child.parameters():
            param.requires_grad = False
'''
model = TripletNet(resnet)
print(model)
if cuda:
    model.cuda()

#model.embedding_net.fc.weight.requires_grad = False
#model.embedding_net.fc.bias.requires_grad = False
'''
for child in model.children():
    count += 1
i = 0
for child in model.children():
    if i < count:
        for param in child.parameters():
            param.requires_grad = False
    i += 1
'''
loss_fn = TripletLoss(margin)
lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 30, gamma=0.1, last_epoch=-1)
n_epochs = 1000000
log_interval = 1
fit(triplet_train_loader, triplet_test_loader, test_loader, ref_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)