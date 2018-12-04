import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from tqdm import tqdm

import torch.nn.functional as F
import torch.optim as optim

from trainer import test, train
from utilites import conv_output_shape, calc_params, flat_shape, GenericEncoder, flatten


class MyClassifierNet(nn.Module):
    def __init__(self, input_shape, num_classes, fc_total=128, activation=F.relu):
        super().__init__()

        self.conv_layers = GenericEncoder(input_shape)
        self.activation = activation

        final_shape = flat_shape(self.conv_layers.output_shape)

        self.fc = nn.Linear(final_shape, fc_total)
        self.to_classes = nn.Linear(fc_total, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.activation(self.fc(flatten(x)))
        x = self.to_classes(x)
        return x




BATCH_SIZE = 128
LEARNING_RATE = 0.01
EPOCHS = 10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)
train_batches = len(trainset) // BATCH_SIZE

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

print("Shape: %s" % list(images.size()))

input_shape = images.size()[1:]
num_classes = len(classes)
print("Shape: %s, %s" % (input_shape, num_classes))

model_args = [input_shape, num_classes]

net = MyClassifierNet(*model_args)
print(net)






device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)


train(net, optimizer, criterion, device, trainloader, trainset, BATCH_SIZE, EPOCHS)

print('Finished Training')

dataiter = iter(testloader)
images, labels = dataiter.next()
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

images, labels = images.to(device), labels.to(device)

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

test(net, device, testloader, classes)


print("saving...")
model_path = "saved_nets/tut_class_cifar10.mod"
torch.save(net.state_dict(), model_path)
print("Saved!")

the_model = MyClassifierNet(*model_args)
the_model.load_state_dict(torch.load(model_path))

print(the_model)

net = the_model
net.to(device)


test(net, device, testloader, classes)





