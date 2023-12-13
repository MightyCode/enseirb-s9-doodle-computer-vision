class DatasetUtils:
    @staticmethod
    def get_n_first_for_label(dataset, label, n=1):
        result = []

        index = 0

        while len(result) < n and index < len(dataset):
            image, image_label = dataset[index]
            if image_label == label:
                result.append(image)

            index += 1

        return result

    @staticmethod
    def get_n_first_random(dataset, n=1):
        return DatasetUtils.get_n_first_for_label(dataset, dataset[0][1], n)
