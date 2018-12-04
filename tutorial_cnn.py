import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    This is actually LeNet with relu activations
    """

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)


params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
# print(params[0][0][0][0])

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
print(out.grad)


net.zero_grad()
out.backward(torch.randn(1, 10))

print(out)
print(out.grad)


output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

print("out: %s" % out.grad)
print("loss", loss.grad)
net.zero_grad()     # zeroes the gradient buffers of all parameters
print("out: %s" % out.grad)
print("loss", loss.grad)

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

# has no gradient, none has been calculated

loss.backward()
# gradient is calculated in backward
print("out: %s" % out.grad)
print("loss", loss.grad)

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


print()
print("Doing SGD!")
print("Before:")
print(net.conv1.bias)
learning_rate = 0.01
for f in net.parameters():
    # for each parameter, subtract the gradient * the learning rate
    # f.data is the actual tensor with the data. f.grad is another tensor. tensors have data...
    # I think it's an infinite loop. doesn't matter, probably. I can try without that .data on the end.
    # f.data.sub_(f.grad.data * learning_rate)
    # ayyy it worked! Original is above, below is after that change
    f.data.sub_(f.grad * learning_rate)

print("After:")
print(net.conv1.bias)

# time to do their optimizers...

import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update

