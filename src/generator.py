import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import os

from utils.PytorchUtils import PytorchUtils

class ImageGenerator():
    def __init__(self, classes, device, model):
        self.classes = classes
        self.nb_classes = len(classes)

        self.model = model

        self.device = device

    """ 
    Generate mean encoded information for each class
        information contain the mean, the var
    Add one additional information for all classes
    """
    def generate_mean_encoded_information(self, images_set):
        mean_encoded_information = []

        if self.model.latent_type == "convolutional":
            mean_vectors_size = (self.model.architecture[-1],
                    int(self.model.width/((len(self.model.architecture)-2)*2)),
                    int(self.model.width/((len(self.model.architecture)-2)*2)))
        elif self.model.latent_type == "vector":
            mean_vectors_size = self.model.architecture[-1]
        elif self.model.latent_type == "convolutional-variational":
            mean_vectors_size = self.model.lantent_dim
        else:
            raise("Latent type not supported")

        count_classes_number = [0] * (self.nb_classes + 1)

        for i in range(self.nb_classes + 1 ):
            mean_encoded_information.append({
                "mean": np.zeros(mean_vectors_size),
                "var": np.zeros(mean_vectors_size)
            })

        self.model.eval()
        with torch.no_grad():
            for batch in images_set:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)

                pack = self.model(images, labels=labels)
                encoded = pack["encoded"]


                encoded_np = encoded.cpu().detach().numpy()

                for i in range(len(images)):
                    print(mean_encoded_information[labels[i]]["mean"].shape, encoded_np[i].shape, encoded_np.shape)
                    mean_encoded_information[labels[i]]["mean"] += encoded_np[i]
                    count_classes_number[labels[i]] += 1

                    mean_encoded_information[self.nb_classes]["mean"] += encoded_np[i]
                    count_classes_number[self.nb_classes] += 1

                    if "encoded_before" in pack:
                        if not "mean_before" in mean_encoded_information[labels[i]].keys():
                            mean_encoded_information[labels[i]]["mean_before"] = np.zeros(mean_vectors_size)
                            mean_encoded_information[labels[i]]["var_before"] = np.zeros(mean_vectors_size)

                        mean_encoded_information[labels[i]]["mean_before"] += pack["encoded_before"][i].cpu().detach().numpy()

                        if not "mean_before" in mean_encoded_information[self.nb_classes].keys():
                            mean_encoded_information[self.nb_classes]["mean_before"] = np.zeros(mean_vectors_size)
                            mean_encoded_information[self.nb_classes]["var_before"] = np.zeros(mean_vectors_size)
                        
                        mean_encoded_information[self.nb_classes]["mean_before"] += pack["encoded_before"][i].cpu().detach().numpy()

                    
            for i in range(self.nb_classes + 1):
                mean_encoded_information[i]["mean"] = mean_encoded_information[i]["mean"] / count_classes_number[i]

                if i < self.nb_classes:
                    name = self.classes[i]
                else:
                    name = "all"
            
                embed_vector = "mean_before" in mean_encoded_information[i].keys()

                print(f'Class {name} range of mean encoded {"embed" if embed_vector else ""} vector: [{mean_encoded_information[i]["mean"].min()},', end="")
                print(f'{mean_encoded_information[i]["mean"].max()}]')

                if "mean_before" in mean_encoded_information[i]:
                    mean_encoded_information[i]["mean_before"] = mean_encoded_information[i]["mean_before"] / count_classes_number[i]

                    print(f'Class {name} range of mean encoded vector: [{mean_encoded_information[i]["mean"].min()},', end="")
                    print(f'{mean_encoded_information[i]["mean"].max()}]')

            # compute the var 

            for batch in images_set:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)

                pack = self.model(images, labels=labels)
                encoded = pack["encoded"]

                encoded_np = encoded.cpu().detach().numpy()

                for i in range(len(images)):
                    mean_encoded_information[labels[i]]["var"] += np.square(encoded_np[i] - mean_encoded_information[labels[i]]["mean"])
                    mean_encoded_information[self.nb_classes]["var"] += np.square(encoded_np[i] - mean_encoded_information[self.nb_classes]["mean"])

                    if "encoded_before" in pack:
                        mean_encoded_information[labels[i]]["var_before"] += np.square(pack["encoded_before"][i].cpu().detach().numpy() - mean_encoded_information[labels[i]]["mean_before"])
                        mean_encoded_information[self.nb_classes]["var_before"] += np.square(pack["encoded_before"][i].cpu().detach().numpy() - mean_encoded_information[self.nb_classes]["mean_before"])

            for i in range(self.nb_classes + 1):
                mean_encoded_information[i]["var"] = mean_encoded_information[i]["var"] / count_classes_number[i]
                mean_encoded_information[i]["var"] = np.sqrt(mean_encoded_information[i]["var"])

                if "mean_before" in mean_encoded_information[i]:
                    mean_encoded_information[i]["var_before"] = mean_encoded_information[i]["var_before"] / count_classes_number[i]
                    mean_encoded_information[i]["var_before"] = np.sqrt(mean_encoded_information[i]["var_before"])

        return mean_encoded_information
    
    """
    Tensor mean and var vector have the same shape
    Use gaussian distribution to generate alternative vector
    """
    def create_alternative_version(self, mean_vector, var_vector, weight=0.1):
        alternative_vector = np.zeros(mean_vector.shape)

        for i in range(len(mean_vector)):
            alternative_vector[i] = np.random.normal(mean_vector[i], weight * var_vector[i])

        return alternative_vector

    """
    Image size means that the model is not convolutional
    Add embedding if labbel
    """
    def generate_images_for_vectors(self, mean_encoded_vectors, labels=None, image_size=None):
        generated_images = []

        decoder = self.model.decoder

        decoder.eval()
        with torch.no_grad():
            for i in range(len(mean_encoded_vectors)):
                mean_vector = mean_encoded_vectors[i]
                double_mean_vector = np.array([mean_vector]).astype(np.float32)
                mean_vector_torch = torch.from_numpy(double_mean_vector).to(self.device)

                if labels is not None:
                    label = torch.tensor(labels[i]).to(self.device).unsqueeze(0)

                    embedding = self.model.get_embed(label).squeeze()
                    mean_vector_torch = self.model.add_class_to_encoded(mean_vector_torch, embedding)

                decoded = decoder(mean_vector_torch).squeeze()

                result = decoded.cpu().detach().numpy()

                if image_size is not None:
                    generated_images.append(result.reshape(image_size[1], image_size[0]))
                else:
                    generated_images.append(result)
            
        return generated_images


    def show_generated_images_per_mean_vectors(self, mean_encoded_vectors, labels=None, image_size=None, title=None):
        generated_images = self.generate_images_for_vectors(mean_encoded_vectors, labels=labels, image_size=image_size)

        num_cols = 4 if len(mean_encoded_vectors) >= 4 else len(mean_encoded_vectors)
        num_rows = math.ceil(len(mean_encoded_vectors) / num_cols)

        _, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))

        for i in range(len(mean_encoded_vectors)):
            row_index = i // num_cols
            col_index = i % num_cols
            axes[row_index, col_index].imshow(generated_images[i], cmap='gray')
            axes[row_index, col_index].axis('off')
            axes[row_index, col_index].set_title(self.classes[i] if i < len(self.classes) else "all")

        plt.tight_layout()
        plt.suptitle(title if title else 'Generated images')

        plt.subplots_adjust(top=0.9)

        plt.show()

    
    def show_generated_images_per_vector(self, vectors, labels=None, titles=None, image_size=None, title=None):
        generated_images = self.generate_images_for_vectors(vectors, labels=labels, image_size=image_size)

        self.show_generated_images(generated_images, titles=titles, title=title)

    def show_generated_images(self, images, titles=None, title=None):
        num_cols = 4 if len(images) >= 4 else len(images)
        num_rows = math.ceil(len(images) / num_cols)

        _, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))

        for i in range(len(images)):
            row_index = i // num_cols
            col_index = i % num_cols
            axes[row_index, col_index].imshow(images[i], cmap='gray')
            axes[row_index, col_index].axis('off')
            if titles and len(titles) > i:
                axes[row_index, col_index].set_title(titles[i], fontsize=10)

        plt.tight_layout()
        plt.suptitle('Generated images' if title is None else title)

        plt.subplots_adjust(top=0.9)

        plt.show()



    """
    Interpolate trough n mean vectors with interpolation_number between each mean vectors
    Add embedding if labels
    Save the images if save_path
    """
    def interpolate_vectors(self, vectors, labels=None, interpolation_number=10, save_path=None, image_size=None):
        interpolated_vectors = []
        vector_labels = None if labels is None else []

        for i in range(len(vectors) - 1):
            
            vector_from = vectors[i]
            vector_to = vectors[i + 1]

            if labels is not None:
                label = labels[i]

            for j in range(interpolation_number):
                interpolated_vectors.append(
                     PytorchUtils.tensor_to_numpy(vector_from + (j / (interpolation_number - 1)) * (vector_to - vector_from))
                     )

                if labels is not None:
                    vector_labels.append(label)

        print(len(interpolated_vectors))
        
        images = self.generate_images_for_vectors(interpolated_vectors, 
                                                  labels=vector_labels, image_size=image_size)

        print(images[0].shape)

        if save_path is not None:
            # check if path exist
            if os.path.exists(save_path):
                # clear the folder
                for file in os.listdir(save_path):
                    os.remove(f'{save_path}{file}')
            else:
                os.makedirs(save_path)
            

            for i in range(len(images)):
                # create 01, 02, 03 names
                plt.imsave(f'{save_path}image-{(i + 1)}.png', images[i], cmap='gray')

        return images

                