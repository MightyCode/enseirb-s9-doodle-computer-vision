from torch.utils.data import Dataset
import numpy as np
from dataset_creation.ImageCreation import ImageCreation 

class StrokeImageDataset(Dataset):
    def __init__(self, data, labels, width, height, stroke_size, factor, reshape=True, normalize=False):
        super().__init__()

        self.stroke_size = stroke_size
        self.factor = factor

        self.reshape = reshape
        self.normalize = normalize

        self.width = width
        self.height = height

        self.data = np.array(data)
        self.labels = np.array(labels)

    def reshape_image(self, img : np.ndarray):
        return img.reshape(self.width * self.height)

    def normalize_image(self, img : np.ndarray):
        return img.astype(np.float32) / 255.0

    def __getitem__(self, key: int):
        assert key < self.__len__()

        img = self.data[key]

        created_image = ImageCreation.createImage(img, (self.width, self.height), self.stroke_size, self.factor)

        if not self.reshape:
            created_image = self.reshape_image(created_image)

        if self.normalize:
            created_image = self.normalize_image(created_image)

        # convert the image width x height to width*height

        label = self.labels[key]

        return created_image, label

    def convert_to_img(self, img):
        return img.reshape(self.width, self.height)

    def get_image_2d(self, key: int):
        return self[key][0].reshape(self.width, self.height)
    
    def __len__(self):
        return len(self.data)