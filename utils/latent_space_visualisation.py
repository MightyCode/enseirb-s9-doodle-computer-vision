from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt

def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def tensor_to_img(tensor, width, height):
    return tensor_to_numpy(tensor).reshape((width, height))

def latent_space_visualization(model, valid_loader, use_embedding=False):
    model.eval()
    pca = PCA(n_components=2)

    points = []
    label_idcs = []
    for data in valid_loader:
        inputs, labels = data
        inputs, labels = inputs.to(model.device), labels.to(model.device)
            
        pack = model(inputs, labels)
        proj = pack["encoded_before" if use_embedding else "encoded"]
        for i in range(inputs.size(0)):
            encoded = tensor_to_numpy(proj[i])

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
    plt.show()