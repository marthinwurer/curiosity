import random

import matplotlib.pyplot as plt
import torch
from torch import optim, nn
from torchvision import transforms

from autoencoder import ProGANAutoencoder, VGImagesDataset, VG_PATH
from torch.utils.data import Dataset
from training_state import Hyperparameters, TrainingState


# noinspection PyPep8Naming
def main():

    filename = "saved_nets/autoencoder_cp_10.tar"

    BATCH_SIZE = 8
    LEARNING_RATE = 0.0001
    EPOCHS = 1
    MOMENTUM = 0.9
    IN_POWER = 8

    in_dim = 2 ** IN_POWER

    hyper = Hyperparameters(BATCH_SIZE, LEARNING_RATE)

    net_transform = transforms.Compose([
        transforms.Resize(in_dim),
        transforms.RandomCrop(in_dim, pad_if_needed=True),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = VGImagesDataset(root_dir=VG_PATH, transform=net_transform)

    net = ProGANAutoencoder(512, IN_POWER, 2)
    print(net)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)

    optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


    train_state = TrainingState(net, optimizer, hyper)
    train_state.load_state(filename)

    view_samples(train_state.model, dataset, device)
    # view_samples(None, dataset)


def view_samples(model: nn.Module, dataset: Dataset, device):

    # generate 4 images
    images = []
    samples = []
    for i in range(4):
        index = random.randrange(len(dataset))
        image, _ = dataset[index]
        print(image)
        encode = image.unsqueeze(0).to(device)
        decoded = model.forward(encode)[0].squeeze().detach().permute(1, 2, 0).cpu().numpy()
        image = image.permute(1, 2, 0)
        images.append(image)
        samples.append(decoded)
        print(decoded)

    f, axis_array = plt.subplots(2, 4)
    axis_array[0, 0].imshow(images[0])
    axis_array[0, 1].imshow(images[1])
    axis_array[1, 0].imshow(images[2])
    axis_array[1, 1].imshow(images[3])
    axis_array[0, 2].imshow(samples[0])
    axis_array[0, 3].imshow(samples[1])
    axis_array[1, 2].imshow(samples[2])
    axis_array[1, 3].imshow(samples[3])
    plt.show()


if __name__ == "__main__":
    main()
