import math
import torch

from tqdm import tqdm


def train(net, optimizer, criterion, device, trainloader, trainset, batch_size, epochs):

    train_batches = math.ceil(len(trainset) / batch_size)
    running_loss = 0.0
    for epoch in range(epochs):  # loop over the dataset multiple times

        loss_steps = 8000/batch_size

        with tqdm(enumerate(trainloader, 0), total=train_batches, unit="batch") as t:
            for i, data in t:
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % loss_steps == 0:    # print every 2000 mini-batches
                    string = '[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / loss_steps)
                    t.set_postfix_str(string)
                    running_loss = 0.0


def test(net, device, testloader, classes):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    class_correct = list(0. for _ in range(len(classes)))
    class_total = list(0. for _ in range(len(classes)))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
