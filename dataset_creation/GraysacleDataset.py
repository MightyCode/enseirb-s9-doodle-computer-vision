from torch.utils.data import Dataset, DataLoader
import numpy as np

class GrayscaleDataset(Dataset):
    def __init__(self, data, labels, width, height):
        super().__init__()

        self.width = width
        self.height = height

        self.data=np.array(data)
        self.labels=np.array(labels)

        self.reshaped_data = [self.reshape_image(img) for img in data]

    def reshape_image(self, img):
        return img.reshape(self.width, self.height)
        

    def __getitem__(self, key: int):
        assert key < self.__len__()

        img = self.reshaped_data[key]
        label = self.labels[key]

        return img, label

    
    def __len__(self):
        return len(self.reshaped_data)