import matplotlib.pyplot as plt
import random

def plot_random_images(set, classes):
    """
    plot random images from the set given into 4x2 subplot
    """
    # Show some images
    numb_rows = 2
    numb_cols = 4

    _, axes = plt.subplots(numb_rows, numb_cols, figsize=(5*numb_cols, 5*numb_rows))
    for i in range(numb_rows*numb_cols):
        index = random.randint(0, len(set)-1)
        ax = axes[i//numb_cols, i%numb_cols]

        img, label_index = set[index]

        ax.imshow(set.convert_to_img(img), cmap='gray')
        ax.set_title(classes[label_index], fontsize=25)

        # no ticks
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()