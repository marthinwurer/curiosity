from torch.utils.data import Dataset
from PIL import Image


class ImageListDataset(Dataset):
    """
    Loads a dataset from a text file with a list of images.
    """
    def __init__(self, path, transform, resolution=8):
        # load the paths from the text file
        with open(path) as f:
            image_paths = []
            for line in f:
                line = line.strip()
                # good enough for now, make an issue if this fails.
                if line.lower().endswith(('gif', 'jpeg', 'jpg',  'png')):
                    image_paths.append(line)

        self.image_paths = image_paths
        self.transform = transform
        self.resolution = resolution

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)

        return img