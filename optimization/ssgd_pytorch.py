from __future__ import print_function
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Barrier
from torchvision import datasets, transforms
from torch.utils.data import Subset
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F


batch_size = 64 # input batch size for training
test_batch_size = 1000 # input batch size for testing
epochs = 3 # number of global epochs to train
lr = 0.01 # learning rate
momentum = 0.5 # SGD momentum
seed = 1 # random seed
log_interval = 10 # how many batches to wait before logging training status
n_workers = 4 # how many training processes to use
cuda = True # enables CUDA training
mps = False # enables macOS GPU training


class CustomSubset(Subset):
    '''A custom subset class with customizable data transformation'''
    def __init__(self, dataset, indices, subset_transform=None):
        super().__init__(dataset, indices)
        self.subset_transform = subset_transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.subset_transform:
            x = self.subset_transform(x)
        return x, y   

    def __len__(self):
        return len(self.indices)

    
def dataset_split(dataset, n_workers):
    n_samples = len(dataset)
    n_sample_per_workers = n_samples // n_workers
    local_datasets = []
    for w_id in range(n_workers):
        if w_id < n_workers - 1:
            local_datasets.append(CustomSubset(dataset, range(w_id * n_sample_per_workers, (w_id + 1) * n_sample_per_workers)))
        else:
            local_datasets.append(CustomSubset(dataset, range(w_id * n_sample_per_workers, n_samples)))
    return local_datasets    


def pull_down(global_W, local_Ws, n_workers):
    # pull down global model to local
    for rank in range(n_workers):
        for name, value in local_Ws[rank].items():
            local_Ws[rank][name].data = global_W[name].data 


def aggregate(global_W, local_Ws, n_workers):
    # init the global model
    for name, value in global_W.items():
        global_W[name].data  = torch.zeros_like(value)
        
    for rank in range(n_workers):
        for name, value in local_Ws[rank].items():
            global_W[name].data += value.data

    for name in local_Ws[rank].keys():
        global_W[name].data /= n_workers
        
        
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
    
        
def train_epoch(epoch, rank, local_model, device, dataset, synchronizer, dataloader_kwargs):
    torch.manual_seed(seed + rank)
    train_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=momentum)

    local_model.train()
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = local_model(data.to(device))
        loss = F.nll_loss(output, target.to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                pid, epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
    # synchronizer.wait()
    
    
def test(epoch, model, device, dataset, dataloader_kwargs):
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
    print('\nTest Epoch: {} Global loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch + 1, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))  
    
    
if __name__ == "__main__":
    use_cuda = cuda and torch.cuda.is_available()
    use_mps = mps and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_dataset = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True,
                       transform=transform)
    local_train_datasets = dataset_split(train_dataset, n_workers)    

    kwargs = {'batch_size': batch_size,
              'shuffle': True}
    if use_cuda:
        kwargs.update({'num_workers': 1, # num_workers to load data
                       'pin_memory': True,
                      })

    torch.manual_seed(seed)
    mp.set_start_method('spawn', force=True) 
    # Very important, otherwise CUDA memory cannot be allocated in the child process

    local_models = [Net().to(device) for i in range(n_workers)]
    global_model = Net().to(device)
    local_Ws = [{key: value for key, value in local_models[i].named_parameters()} for i in range(n_workers)]
    global_W = {key: value for key, value in global_model.named_parameters()}
    
    synchronizer = Barrier(n_workers)
    for epoch in range(epochs):
        for rank in range(n_workers):
            # pull down global model to local
            pull_down(global_W, local_Ws, n_workers)
            
            processes = []
            for rank in range(n_workers):
                p = mp.Process(target=train_epoch, args=(epoch, rank, local_models[rank], device,
                                                local_train_datasets[rank], synchronizer, kwargs))
                # We first train the model across `num_processes` processes
                p.start()
                processes.append(p)
                            
            for p in processes:
                p.join()
    
        aggregate(global_W, local_Ws, n_workers)

        # We test the model each epoch
        test(epoch, global_model, device, test_dataset, kwargs)
    # Test result for synchronous training：Test Epoch: 3 Global loss: 0.0732, Accuracy: 9796/10000 (98%)
    # Test result for asynchronous training：Test Epoch: 3 Global loss: 0.0742, Accuracy: 9789/10000 (98%)
  