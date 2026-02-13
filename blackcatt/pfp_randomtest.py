
from matplotlib import pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from random import sample
from torchvision.transforms import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, Lambda
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

## Aux functions and classes ##

def ResNet18():
    return ResNet(ResidualBlock)

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=100):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)        
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)        
        self.fc = nn.Linear(512, num_classes)
        
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            try:
                images = batch["img"].to(device)
            except:
                images = batch["image"].to(device)
            try:
                labels = batch["label"].to(device)
            except:
                labels = batch["fine_label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
    
def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR100 data."""
    # Only initialize `FederatedDataset` once
    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="uoft-cs/cifar100",
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(partition_id)
    centralized_testset = fds.load_split("test")

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [RandomCrop(32, padding=4),RandomHorizontalFlip(),ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    centralized_testset = centralized_testset.with_transform(apply_transforms)

    trainloader = DataLoader(partition_train_test["train"], batch_size=128, shuffle=True)
    testloader = DataLoader(centralized_testset, batch_size=128)
    return trainloader, testloader


def tardos_score(x,y,p):
  # Calculate the Tardos score for a secuence of outputs 
  # and a specific client vector
  t_score = 0
  for i_p in range(p.shape[0]):
    if y[i_p] == x[i_p]:
      t_score += np.sqrt((1-p[i_p,y[i_p]])/p[i_p,y[i_p]])
    else:
      t_score -= np.sqrt((p[i_p,y[i_p]])/(1-p[i_p,y[i_p]]))
  return t_score


def tardos_accusation(y,vectors,p_secret,tau,pfp=0.000001):
  t_score = np.zeros(vectors.shape[0])

  for mi in range(vectors.shape[1]):
    # Calculate threshold according to Skoric and Oosterwijk 2012
    a = 1/(2*np.log(pfp))
    b = 1/(3*np.sqrt(tau))
    c = mi+1
    disc = b*b-4*a*c
    if disc >= 0:
      x1 = (-b+np.sqrt(disc))/(2*a)
      x2 = (-b-np.sqrt(disc))/(2*a)
      if x1 > 0 and x2 > 0:
        Z = min([x1,x2])
      elif x1 > 0:
        Z = x1
      elif x2 > 0:
        Z = x2
    
    # Calculate the Tardos score for each client
    for client_index in range(vectors.shape[0]):
      t_score[client_index] += tardos_score(vectors[client_index,mi-1:mi],y[mi-1:mi],p_secret[mi-1:mi])
    
    # Accuse if the score is above the threshold
    if max(t_score) > Z:
      return np.argmax(t_score) , mi+1
    
  return -1 , mi+1

# Test FPR of Tardos scheme with random independent models

# WM parameters
m = 250
n_users = 60
clients_tardos_q = np.loadtxt("wm_constants/" + 'clients_tardos_q_100_k0.5.csv').astype(int)[:n_users,:m]
p_secret = np.loadtxt("wm_constants/" + 'p_secret_100_k0.5.csv').astype(float)[:m,:]
tau =  0.001

# Path to triggers
index = 431
triggers_path = './blackcatt/triggers_'+str(index)+'.npy'

net = ResNet18()
net.to(device)
net.eval()

fn = []
n_collusions = 3000
n_epochs = 100
pfp = 0.01

# All CIFAR100 data for training random models
trainloader, valloader = load_data(0, 1)

exp_run = np.random.randint(0,10000)

df_all = pd.DataFrame({"epochs":[],"final_acc":[],"m_needed":[],"fn":[],"fp":[]})

print(triggers_path)

triggers = np.load(triggers_path)
triggers_transforms = Compose(
    [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
triggers = torch.stack([triggers_transforms(Image.fromarray(trigger.astype(np.uint8))) for trigger in triggers]).float().to(device)

for i in tqdm(range(n_collusions)):     
    fn = []
    fp = []
    m_needed = []
    net = ResNet18()
    net.to(device)
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    # Train a random model copy
    for epoch in range(n_epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["fine_label"]
            inputs, targets = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()    
    try:
        net.eval()
        test_loss, test_acc = test(net, valloader, device)
        print(test_acc)
        output = net(triggers)

        y = torch.argmax(output, dim=1).cpu().numpy()
        
        tp, t_s = tardos_accusation(y,vectors=clients_tardos_q, p_secret=p_secret, tau=tau,pfp=pfp)
        if tp==-1:
            # No accusation (correct behavior)
            fn.append(0)
            fp.append(0)
            m_needed.append(triggers.shape[0])
        else:
            # False positive
            fn.append(0)
            fp.append(1)
            m_needed.append(t_s)
    except:
        pass

    df = pd.DataFrame({"epochs":[n_epochs],"final_acc":[test_acc],"m_needed":m_needed,"fn":fn,"fp":fp})
    df_all = pd.concat([df_all,df])

    df_all.to_csv("FL_models/df_all_pfp"+str(pfp)+"_"+str(index)+"_cNoWM_"+str(exp_run)+".csv",index=False)
