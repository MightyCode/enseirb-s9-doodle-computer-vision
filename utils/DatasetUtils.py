class DatasetUtils:
    @staticmethod
    def get_n_first_for_label(dataset, label, n=1, start_index=0):
        images = []
        labels = []
        indices = []

        index = start_index

        while len(images) < n and index < len(dataset):
            image, image_label = dataset[index]
            if image_label == label:
                images.append(image)
                labels.append(image_label)
                indices.append(index)

            index += 1

        return images, labels, indices

    @staticmethod
    def get_n_first_random(dataset, n=1, start_index=0):
        return DatasetUtils.get_n_first_for_label(dataset, dataset[0][1], n, start_index)
