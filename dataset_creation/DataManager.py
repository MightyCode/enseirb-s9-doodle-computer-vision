import numpy as np
import random

class DataManager():
    def split_data(self, split: float, data):
        """
        splits the data into training and validation data and labels


        output specs : 
        len(training_data) = split*len(data)
        len(validation_data) = (1-split)*len(data)
        """

        training_data, validation_data = [], []
        training_labels, validation_labels = [], []

        for class_data, label in zip(data, np.arange(start=0, stop=8)):
            np.random.shuffle(class_data)

            split_idx=int(split*len(class_data))
            
            for img in class_data[:split_idx]:
                training_data.append(img.astype(np.float32))
                training_labels.append(label)

            for img in class_data[split_idx:]:
                validation_data.append(img.astype(np.float32))
                validation_labels.append(label)

        assert len(training_data) == len(training_labels)
        assert len(validation_data) == len(validation_labels)

        return training_data, training_labels, validation_data, validation_labels

    def shuffle_dataset(self, data, labels):
        """
        shuffles data and labels according to the same permutation
        """
        assert len(data) == len(labels)

        combined_lists = list(zip(data, labels))

        permutation = list(range(len(combined_lists)))
        random.shuffle(permutation)

        shuffled_lists = [combined_lists[i] for i in permutation]
        shuffled_data, shuffled_labels = zip(*shuffled_lists)

        return shuffled_data, shuffled_labels