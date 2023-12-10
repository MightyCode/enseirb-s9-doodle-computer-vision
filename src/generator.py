import numpy as np
import torch
import matplotlib.pyplot as plt

class ImageGenerator():
    def __init__(self, classes, device, model):
        self.classes = classes
        self.nb_classes = len(classes)

        self.model = model

        self.device = device
        pass

    def generate_mean_encoded_vectors_per_classes(self, images_set):
        mean_encoded_vectors = []
        mean_vectors_size = (self.model.architecture[-1],
                            int(self.model.width/((len(self.model.architecture)-2)*2)),
                            int(self.model.width/((len(self.model.architecture)-2)*2)))
        
        count_classes_number = [0] * self.nb_classes

        for i in range(self.nb_classes):
            mean_encoded_vectors.append(np.zeros(mean_vectors_size))

        for batch in images_set:
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)

            encoded, _ = self.model(images)
            encoded_np = encoded.cpu().detach().numpy()

            for i in range(len(images)):
                mean_encoded_vectors[labels[i]] += encoded_np[i]
                count_classes_number[labels[i]] += 1
                
        for i in range(self.nb_classes):
            mean_encoded_vectors[i] = mean_encoded_vectors[i] / count_classes_number[i]

            print(f'Class {self.classes[i]} range of mean encoded vector: [{mean_encoded_vectors[i].min()}, {mean_encoded_vectors[i].max()}]')

        return mean_encoded_vectors
    
    def generate_noised_mean_vectors(self, mean_encoded_vectors, noise):
        alternative_versions = []
        for mean_encoded_vector in mean_encoded_vectors:
            alternative_versions.append(self.create_alternative_version(mean_encoded_vector, noise))
        
        return alternative_versions
    
    def create_alternative_version(self, mean_vector, weight=0.1):
        alternative_mean_vector = mean_vector.copy()
        # Vector is composed of float values 
        # use gaussian distribution to generate altertivate vector based on mean one

        for i in range(len(mean_vector)):
            alternative_mean_vector[i] = np.random.normal(mean_vector[i], weight)
        
        return alternative_mean_vector

    def generate_images_for_mean_vectors(self, mean_encoded_vectors):
        generated_images = []

        decoder = self.model.decoder
        for i in range(self.nb_classes):
            mean_vector = mean_encoded_vectors[i]
            double_mean_vector = np.array([mean_vector]).astype(np.float32)
            mean_vector_torch = torch.from_numpy(double_mean_vector).to(self.device)

            decoded = decoder(mean_vector_torch).squeeze()

            generated_images.append(decoded.cpu().detach().numpy())

        return generated_images

    def show_generated_images_per_mean_vectors(self, mean_encoded_vectors):
        generated_images = self.generate_images_for_mean_vectors(mean_encoded_vectors)

        num_cols = 4
        num_rows = 2

        _, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))

        for i in range(self.nb_classes):
            row_index = i // num_cols
            col_index = i % num_cols
            axes[row_index, col_index].imshow(generated_images[i], cmap='gray')
            axes[row_index, col_index].axis('off')
            axes[row_index, col_index].set_title(self.classes[i])

        plt.tight_layout()
        plt.suptitle('Generated images')

        plt.subplots_adjust(top=0.9)

        plt.show()