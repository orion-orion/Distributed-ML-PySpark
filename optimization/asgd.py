import os
import threading
from datetime import datetime
import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
from torch import optim
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import Subset


batch_size = 20
n_workers = 5
epochs = 10
seed = 1
log_interval = 10 # how many epochs to wait before logging training status
cuda = True # enables CUDA training
mps = False # enables macOS GPU training
use_cuda = cuda and torch.cuda.is_available()
use_mps = mps and torch.backends.mps.is_available()
if use_cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    

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

class ParameterServer(object):

    def __init__(self, n_workers=n_workers):
        self.model = Net().to(device)
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        self.n_workers = n_workers
        self.curr_update_size = 0
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=32, shuffle=True)


    def get_model(self):
        # TensorPipe RPC backend only supports CPU tensors, 
        # so we move your tensors to CPU before sending them over RPC
        return self.model.to("cpu")

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads):
        self = ps_rref.local_value()
        for p, g in zip(self.model.parameters(), grads):
            p.grad += g
        with self.lock:
            self.curr_update_size += 1
            fut = self.future_model

            if self.curr_update_size >= self.n_workers:
                for p in self.model.parameters():
                    p.grad /= self.n_workers
                self.curr_update_size = 0
                self.optimizer.step()
                self.optimizer.zero_grad()
                fut.set_result(self.model)
                self.future_model = torch.futures.Future()

        return fut

    def evaluation(self):
        self.model.eval()
        self.model = self.model.to(device)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data.to(device))
                test_loss += F.nll_loss(output, target.to(device), reduction='sum').item() # sum up batch loss
                pred = output.max(1)[1] # get the index of the max log-probability
                correct += pred.eq(target.to(device)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        print('\nTest result - Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(self.test_loader.dataset), 100. * correct / len(self.test_loader.dataset)))  


class Trainer(object):

    def __init__(self, ps_rref):
        self.ps_rref = ps_rref
        self.model = Net().to(device) 

    def train(self, train_dataset):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model = self.ps_rref.rpc_sync().get_model().cuda()
        pid = os.getpid()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                output = model(data.to(device))
                loss = F.nll_loss(output, target.to(device))
                loss.backward()
                model = rpc.rpc_sync(
                    self.ps_rref.owner(),
                    ParameterServer.update_and_fetch_model,
                    args=(self.ps_rref, [p.grad for p in model.cpu().parameters()]),
                ).cuda()
                if batch_idx % log_interval == 0:
                    print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        pid, epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
            


def run_trainer(ps_rref, train_dataset):
    trainer = Trainer(ps_rref)
    trainer.train(train_dataset)


def run_ps(trainers):
    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    local_train_datasets = dataset_split(train_dataset, n_workers)    
    
    
    print(f"{datetime.now().strftime('%H:%M:%S')} Start training")
    ps = ParameterServer()
    ps_rref = rpc.RRef(ps)
    futs = []
    for idx, trainer in enumerate(trainers):
        futs.append(
            rpc.rpc_async(trainer, run_trainer, args=(ps_rref, local_train_datasets[idx]))
        )

    torch.futures.wait_all(futs)
    print(f"{datetime.now().strftime('%H:%M:%S')} Finish training")
    ps.evaluation()
    # Test result - Accuracy: 9696/10000 (97%)

def run(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options=rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=0  # infinite timeout
     )
    if rank == 0:
        rpc.init_rpc(
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_ps([f"trainer{r}" for r in range(1, world_size)])
    else:
        rpc.init_rpc(
            f"trainer{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        # trainer passively waiting for ps to kick off training iterations

    # block until all rpcs finish
    rpc.shutdown()


if __name__=="__main__":
    world_size = n_workers + 1
    mp.spawn(run, args=(world_size, ), nprocs=world_size, join=True)
