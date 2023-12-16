import json, os, sys, argparse

from src.InitModel import InitModel
import torch
import torch.nn as nn

import numpy as np  
from torch.utils.data import DataLoader
from dataset_creation.DataManagerNpz import DataManagerNpz
from dataset_creation.StrokeImageDataset import StrokeImageDataset

from src.generator import ImageGenerator

"""
- f -- file, per default: "models.json" will search in resources folder
"""
def create_arg_parse():
    parser = argparse.ArgumentParser(description='Create images from a model')
    parser.add_argument('-f', '--file', type=str, default="models.json", help='JSON file containing models')
    
    return parser


if __name__ == "__main__":
    parser = create_arg_parse()
    args = parser.parse_args()

    with open(os.path.join("resources", args.file), "r") as f:
        models = json.load(f)

    previous_dim = [None, None]
    dataset = None

    """
    INIT DEVICE
    """
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
    
    """
    LOAD DATA
    """
    print("Loading data ... \n")
    resources_folder = 'resources/sketchrnn'
    classes = ['apple', 'golf club', 'hedgehog', 'moon', 'mushroom', 'rain', 'roller coaster', 'squirrel']
    nb_classes = len(classes)
    class_size = {}

    data = []

    for class_name in classes:
        data_class = np.load(os.path.join(resources_folder, 'sketchrnn_' + class_name + '.npz'), allow_pickle=True, encoding="latin1")

        class_size[class_name] = len(data_class)
        data.append(data_class)

    print("Creating datamanager ... \n")
    data_manager = DataManagerNpz()
    training_data, training_labels, _, _, _, _ = data_manager.parse_data(data)

    LEN_SUBSET = 20000
    STROKE_SIZES = {
        128 : 3,
        64 : 2
    }
    IMAGE_FACTOR = 1.1

    for model in models["models"]:
        print(f"Model : {model['model']}, path : {model['path']}")
        parameters = model["parameters"]
        CONV_MODEL_DATA = "conv" in model["model"] or "convolutionnal" in model["model"]
        
        # If image size changes, or if it's the first iteration 
        # dataset must be initialized
        if previous_dim[0] != parameters["image_size"][0] or previous_dim[1] != parameters["image_size"][1]:
            print("Reload dataset for images  : " + str(parameters["image_size"]))
            dataset = StrokeImageDataset(
                data=training_data[:LEN_SUBSET], 
                labels=training_labels[:LEN_SUBSET],
                width=parameters["image_size"][0], height=parameters["image_size"][1], 
                stroke_size=STROKE_SIZES[parameters["image_size"][0]], factor=IMAGE_FACTOR, 
                reshape=CONV_MODEL_DATA, normalize=True)

        CONV_ARCHITECTURE = [1, 32, 16, 8]
        LINEAR_ARCHITECTURE = [parameters["image_size"][0] * parameters["image_size"][1], 
                               parameters["image_size"][0] * parameters["image_size"][1] * 3 // 4]

        autoencoder_model, is_embed_model = InitModel.init_model(
            model["model"], 
            device, 
            parameters["image_size"], parameters["image_size"], classes, 
            CONV_ARCHITECTURE, LINEAR_ARCHITECTURE, parameters["dropout"], parameters["batch_norm"], 
            parameters["rl"] if "rl" in parameters.keys() else 1, 
            parameters["kl"] if "kl" in parameters.keys() else 0,
            verbose=False)
        
        InitModel.print_model_characteristics(autoencoder_model)
        
        optimzer, criterion = InitModel.create_criterion_optimizer(model["model"], autoencoder_model, parameters["lr"])

        path = os.path.join("weigths", model["path"] + ".pt")

        # Load weights
        autoencoder_model.train_autoencoder(dataset, dataset, optimzer, criterion, parameters["epochs"], path=path)

        image_generator = ImageGenerator(classes, device, autoencoder_model)

        """
        DO 
        model \ ->
        one file per model : 
            - train curve
            - examples of reconstruction 
            - latent space (2 versions if embed)
            - mean vectors
            - deviation from the mean by variance
            - for all classes an example of reconstruction by adding the embed of the other classes (if model embed)
            - n -> n interpolation (for all classes)
            - associated gifs
            - interpolation n -> m (for all combinations)
            - associated gifs
            - if embed the following interpolations (before, after, other)
            - associated gifs
        categories \
            -> all the categories listed above with all the templates in the folder
        """

        # output folder
        output_folder = os.path.join("results/images", model["path"])
        # Check if exitst
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Check model and categories folder
        model_folder = os.path.join(output_folder, "model")
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        
        categories_folder = os.path.join(output_folder, "categories")
        if not os.path.exists(categories_folder):
            os.makedirs(categories_folder)

        

        