from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision import datasets, transforms
import os
import torch
import torch.optim as optim
import torch.nn.functional as F


batch_size = 64 # input batch size for training
test_batch_size = 1000 # input batch size for testing
epochs = 10 # number of global epochs to train
lr = 0.01 # learning rate
momentum = 0.5 # SGD momentum
seed = 1 # random seed
log_interval = 10 # how many batches to wait before logging training status
n_workers = 4 # how many training processes to use
cuda = True # enables CUDA training
mps = False # enables macOS GPU training
dry_run = False # quickly check a single pass


def train(rank, model, device, dataset, dataloader_kwargs):
    torch.manual_seed(seed + rank)

    train_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    for epoch in range(1, epochs + 1):
        model.train()
        pid = os.getpid()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.to(device))
            loss = F.nll_loss(output, target.to(device))
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    pid, epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                if dry_run:
                    break


def test(model, device, dataset, dataloader_kwargs):
    torch.manual_seed(seed)
    test_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.to(device))
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item() # sum up batch loss
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Global loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))  
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    
if __name__ == '__main__':
    use_cuda = cuda and torch.cuda.is_available()
    use_mps = mps and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(device)
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    test_dataset = datasets.MNIST('../data', train=False,
                       transform=transform)
    kwargs = {'batch_size': batch_size,
              'shuffle': True}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                      })

    torch.manual_seed(seed)
    mp.set_start_method('spawn', force=True)

    model = Net().to(device)
    model.share_memory() # gradients are allocated lazily, so they are not shared here

    processes = []
    for rank in range(n_workers):
        p = mp.Process(target=train, args=(rank, model, device,
                                           train_dataset, kwargs))
        # We first train the model across `n_workers` processes
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    # Once training is complete, we can test the model
    test(model, device, test_dataset, kwargs)
    # Test set: Global loss: 0.0325, Accuracy: 9898/10000 (99%)