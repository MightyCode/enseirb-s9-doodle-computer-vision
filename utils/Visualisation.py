import matplotlib.pyplot as plt
import random

from utils.PytorchUtils import PytorchUtils

from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt

class Visualisation:
    @staticmethod
    def plot_random_images(dataset, classes, title=None):
        """
        plot random images from the set given into 4x2 subplot
        """
        # Show some images
        numb_rows = 2
        numb_cols = 4

        _, axes = plt.subplots(numb_rows, numb_cols, figsize=(5*numb_cols, 5*numb_rows))
        for i in range(numb_rows*numb_cols):
            index = random.randint(0, len(dataset)-1)
            ax = axes[i//numb_cols, i%numb_cols]

            img, label_index = dataset[index]

            ax.imshow(dataset.convert_to_img(img), cmap='gray')
            ax.set_title(classes[label_index], fontsize=25)

            # no ticks
            ax.set_xticks([])
            ax.set_yticks([])

        if title:
            plt.suptitle(title, fontsize=30)
        plt.tight_layout()

    @staticmethod
    def latent_space_visualization(model, valid_loader, use_embedding=False, path=None):
        model.eval()
        pca = PCA(n_components=2)

        points = []
        label_idcs = []
        for data in valid_loader:
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
                
            pack = model(inputs, labels)
            key_word = "encoded"

            if use_embedding and "encoded_before" in pack.keys():
                key_word = "encoded_before"

            proj = pack[key_word]
            for i in range(inputs.size(0)):
                encoded = PytorchUtils.tensor_to_numpy(proj[i])

                # Reshape to flatten in order to get one dimension
                if len(encoded.shape) > 1:
                    encoded = encoded.reshape(-1)

                points.append(encoded)

                label_idcs.append(labels[i].detach().cpu().numpy())
                
        points = np.array(points)
        points = pca.fit_transform(points)
        
        # Creating a scatter plot
        fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(x=points[:, 0], y=points[:, 1], s=2.0, c=label_idcs, cmap='tab10', alpha=0.9, zorder=2)

        ax.grid(True, color="lightgray", alpha=1.0, zorder=0)

        plt.legend(*scatter.legend_elements(), loc="lower right", title="Classes")
        plt.title("Latent space visualization " + ("using embedding" if use_embedding else ""))

        if path:
            plt.savefig(path)
            plt.clf()
        else:
            plt.show()

    @staticmethod
    def dataset_distribution(classes, dataset):
        """
        returns a dictionnary distribution for each class
        """
        distrib = {i: 0 for i in range(len(classes))}

        for i in range(dataset.__len__()):
            _, label = dataset.__getitem__(i)
            distrib[label]+=1

        return distrib