import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader

train_path = "F:/Project/vegetable-classifiation/Vegetable Images/train"
val_path = "F:/Project/vegetable-classifiation/Vegetable Images/validation"
test_path = "F:/Project/vegetable-classifiation/Vegetable Images/test"

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1) 
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(50 * 53 * 53, 500)
        self.fc2 = nn.Linear(500, 15)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 50 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()

    if batch_idx % 100 == 0:
       print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def validate(model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
  #using gpu if available
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  batch_size = 64

  stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

  #image augmentation ans transformation
  trainTfms = tt.Compose([ tt.RandomHorizontalFlip(), 
                          tt.ToTensor(), 
                          tt.Normalize(*stats,inplace=True),
                          tt.Resize((224,224))
                        ] )
  validTestTfms = tt.Compose([tt.ToTensor(), 
                              tt.Normalize(*stats),
                              tt.Resize((224,224))])

  train_ds = ImageFolder(train_path, trainTfms)
  test_ds = ImageFolder(test_path, validTestTfms)
  valid_ds = ImageFolder(val_path, validTestTfms)

  batch_size = 32

  #Dataloader
  train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
  valid_dl = DataLoader(valid_ds, batch_size, num_workers=3, pin_memory=True)
  test_dl = DataLoader(test_ds, batch_size*2, num_workers=3, pin_memory=True)

  #Model Training,Validation & Evaluation
  model = Network().to(device)
  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

  for epoch in range(2):
    train(model, device, train_dl, optimizer, epoch)
    validate(model, device,valid_dl)
    test(model, device, test_dl)

  # Specify the output directory
  output_dir = "/content/output"
  os.makedirs(output_dir, exist_ok=True)

  #saving model
  model_path = os.path.join(output_dir, "torch_model.pt")
  torch.save(model.state_dict(), model_path)
  print("Model saved successfully at:", model_path)



if __name__ == '__main__':
  main()



