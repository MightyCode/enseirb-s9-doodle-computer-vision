import numpy as np
import random

class DataManagerNpz():
    def parse_data(self, data) -> tuple:
        """
        splits the data into training and validation data and labels


        output specs : 
        len(training_data) = split*len(data)
        len(validation_data) = (1-split)*len(data)
        """

        training_data, validation_data, test_data = np.array([]), np.array([]), np.array([])
        training_labels, validation_labels, test_labels = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

        for i in range(len(data)):
            label = i
            
            training_data = np.append(training_data, data[i]["train"])
            validation_data = np.append(validation_data, data[i]["valid"])
            test_data = np.append(test_data, data[i]["test"])

            training_labels = np.append(training_labels, [label]*len(data[i]["train"]))
            validation_labels = np.append(validation_labels, [label]*len(data[i]["valid"]))
            test_labels = np.append(test_labels, [label]*len(data[i]["test"]))

        assert len(training_data) == len(training_labels)
        assert len(validation_data) == len(validation_labels)
        assert len(test_data) == len(test_labels)

        return training_data, training_labels, validation_data, validation_labels, test_data, test_labels


    def shuffle_dataset(self, data : list, labels: list):
        """
        shuffles data and labels according to the same permutation
        """
        assert len(data) == len(labels)

        permutation = np.random.permutation(len(data))

        data = np.array(data)[permutation]
        labels = np.array(labels)[permutation]

        return data, labels