from torch.utils.data import Dataset, DataLoader
import numpy as np

class GrayscaleDataset(Dataset):
    def __init__(self, data, labels, width, height, reshape=True, normalize=False):
        super().__init__()

        self.reshape = reshape
        self.normalize = normalize

        self.width = width
        self.height = height

        self.data=np.array(data)
        self.labels=np.array(labels)

        if self.reshape:
            self.data = [self.reshape_image(img) for img in self.data]

        if self.normalize:
            self.data = [self.normalize_image(img) for img in self.data]

    def reshape_image(self, img):
        return img.reshape(self.width, self.height)

    def normalize_image(self, img):
        return img/255.0

    def __getitem__(self, key: int):
        assert key < self.__len__()

        img = self.data[key]
        label = self.labels[key]

        return img, label

    def get_image_2d(self, key: int):
        return self.data[key].reshape(self.width, self.height)
    

    
    def __len__(self):
        return len(self.data)