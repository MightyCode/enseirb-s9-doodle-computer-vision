from dataset_creation.DataManagerNpz import DataManagerNpz
from dataset_creation.StrokeImageDataset import StrokeImageDataset

from src.InitModel import InitModel

from utils.ImageGenerator import ImageGenerator
from utils.PytorchUtils import PytorchUtils
from utils.Visualisation import Visualisation
from utils.DatasetUtils import DatasetUtils

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import json, os, shutil, argparse
import torch
import numpy as np  


def return_example_model_file():
    return {
        "models" : [
            {
                "model" : "model-name",
                "path" : "weights/model-weights-name.pt",
                "parameters" : {
                    "lr" : 0,
                    "epochs" : 0,
                    "batch_size" : 0,
                    "image_size" : [0, 0],
                }
            }
        ]
    }

"""
-f -- file, per default: "models.json" will search in resources folder
-r -- replace, option if present that replace the file path wy the example model file, exits the program after
"""
def create_arg_parse():
    parser = argparse.ArgumentParser(description='Create images from a model')
    parser.add_argument('-f', '--file', type=str, default="models.json", help='JSON file containing models')
    parser.add_argument('-r', '--replace', action='store_true', help='Replace the file path by the example model file (in resources/folder)')
    
    return parser


def check_folder(path, clear = False):
    if os.path.exists(path):
        if clear:
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

    else: 
        os.makedirs(path)

def gen_train_curve(autoencoder_model, model_output_folder, categories_folder, model_name):
    print("Creating train curve ... \n")
    autoencoder_model.plot_psnr_ssim(path=os.path.join(model_output_folder, "train_curve.png"))

    categorie_train_curve_folder = os.path.join(categories_folder, "train_curve")
    check_folder(categorie_train_curve_folder)
    # Copy train curve to categories folder with rename
    shutil.copy(os.path.join(model_output_folder, "train_curve.png"),
            os.path.join(categorie_train_curve_folder, model_name + "_train_curve.png"))

def gen_reconstruction(autoencoder_model, train_dataset, test_dataset, model_output_folder, categories_folder, model_name):
    print("Creating reconstruction examples ... \n")
    number_time = 5
    model_reconstruction_folder = os.path.join(model_output_folder, "example_reconstruction")
    check_folder(model_reconstruction_folder)
    categorie_resconstruction_folder = os.path.join(categories_folder, "example_reconstruction")
    check_folder(categorie_resconstruction_folder)

    for i in range(number_time):
        autoencoder_model.show_images(train_dataset, test_dataset, 
                                        path=os.path.join(model_reconstruction_folder, f"reconstruction_examples_{i}.png"))

        shutil.copy(os.path.join(model_reconstruction_folder, f"reconstruction_examples_{i}.png"),
                    os.path.join(categorie_resconstruction_folder, model_name + f"_reconstruction_examples_{i}.png"))

def gen_latent_space(autoencoder_model, training_loaded_set, model_output_folder, categories_folder, model_name, is_embed_model):
    print("Creating latent spaces ... \n")
    Visualisation.latent_space_visualization(autoencoder_model, training_loaded_set, use_embedding=False,
                                                path=os.path.join(model_output_folder, "latent_space.png"))
    
    categorie_latent_space_folder = os.path.join(categories_folder, "latent_space")
    check_folder(categorie_latent_space_folder)

    shutil.copy(os.path.join(model_output_folder, "latent_space.png"),
                os.path.join(categorie_latent_space_folder, model_name + "_latent_space.png"))
    
    if is_embed_model:
        Visualisation.latent_space_visualization(autoencoder_model, training_loaded_set, use_embedding=True,
                                                path=os.path.join(model_output_folder, "latent_space_embed.png"))
        
        shutil.copy(os.path.join(model_output_folder, "latent_space_embed.png"),
                            os.path.join(categorie_latent_space_folder, model_name + "_latent_space_embed.png"))

def gen_mean_vectors(autoencoder_model, training_loaded_set, model_output_folder, categories_folder, model_name, image_generator):
    print("Creating mean vectors ... \n")

    mean_encoded_information = image_generator.generate_mean_encoded_information(training_loaded_set, verbose=False)

    has_embedding = "mean_before" in mean_encoded_information[0].keys()

    mean_vectors = []
    for i in range(len(mean_encoded_information)):
        mean_vectors.append(mean_encoded_information[i]['mean'])

    image_generator.show_generated_images_per_mean_vectors(
            mean_vectors, 
            title=("Mean embedded latent spaces" if has_embedding else "Mean latent spaces"),
            image_size=(None if autoencoder_model.latent_type == "convolutional" else (WIDTH, HEIGHT)),
            path=os.path.join(model_output_folder, "mean_vectors.png"))
    
    categorie_mean_vectors_folder = os.path.join(categories_folder, "mean_vectors")
    check_folder(categorie_mean_vectors_folder)

    shutil.copy(os.path.join(model_output_folder, "mean_vectors.png"),
                os.path.join(categorie_mean_vectors_folder, model_name + "_mean_vectors.png"))
    
    return mean_encoded_information


def gen_modified_mean_vectors(autoencoder_model, 
                              model_output_folder, categories_folder, model_name, image_generator, classes):
    print("Creating generated modified mean vectors ... \n")
    number_time = 5

    model_mean_vector_folder = os.path.join(model_output_folder, "generated_mean_vectors")
    check_folder(model_mean_vector_folder)

    categorie_mean_vector_folder = os.path.join(categories_folder, "generated_mean_vectors")
    check_folder(categorie_mean_vector_folder)

    for j in range(number_time):
        noised_encoded_vectors = []
        titles = []

        for i in range(len(mean_encoded_information)):
            alternative = image_generator.create_alternative_version(
                    mean_encoded_information[i]['mean'], 
                    mean_encoded_information[i]['var'], 0.6  + 0.6 * (i / number_time))

            noised_encoded_vectors.append(alternative)

            if i < len(classes):
                titles.append(classes[i])
            else:
                titles.append("all")

        image_generator.show_generated_images_per_vector(
            noised_encoded_vectors, 
            titles=titles, 
            title="Generated image usign random", 
            image_size=(None if autoencoder_model.latent_type == "convolutional" else (WIDTH, HEIGHT)),
            path=os.path.join(model_mean_vector_folder, f"generated_mean_vectors_{j}.png"))
        
        shutil.copy(os.path.join(model_mean_vector_folder, f"generated_mean_vectors_{j}.png"),
                    os.path.join(categorie_mean_vector_folder, model_name + f"_generated_mean_vectors_{j}.png"))

def gen_embed_images_with_other_classes(autoencoder_model, 
                                        model_output_folder, categories_folder, model_name, 
                                        image_generator, classes, dataset):
    print("Creating embed images with other classes ... \n")
    model_embed_folder = os.path.join(model_output_folder, "embed_images")
    check_folder(model_embed_folder)

    categorie_embed_folder = os.path.join(categories_folder, "embed_images")
    check_folder(categorie_embed_folder)                

    for j in range(len(classes)):
        image, label, index = DatasetUtils.get_n_first_for_label(dataset, j, 1)
        image = image[0]
        label = label[0]
        index = index[0]

        latent_spaces = []
        labels = []

        image = PytorchUtils.numpy_to_tensor(image, device).unsqueeze(0)
        label = torch.tensor(label).to(device).unsqueeze(0)

        pack = autoencoder_model(image, label)

        encoded_before = PytorchUtils.tensor_to_numpy(pack["encoded_before"])

        for i in range(len(classes)):
            latent_spaces.append(encoded_before)
            labels.append(i)

        image_generator.show_generated_images_per_vector(
            latent_spaces, 
            labels=labels,
            titles=classes, 
            title=f"Generated image on {classes[label]} using embedding", 
            image_size=(None if autoencoder_model.latent_type == "convolutional" else (WIDTH, HEIGHT)),
            path=os.path.join(model_embed_folder, f"embed_images_{j}.png"))
        
        shutil.copy(os.path.join(model_embed_folder, f"embed_images_{j}.png"),
                    os.path.join(categorie_embed_folder, model_name + f"_embed_images_{j}.png"))

def create_folder(model_interpolation_folder, categorie_interpolation_folder, path):
    model_path = os.path.join(model_interpolation_folder, path)
    check_folder(model_path)

    model_path_gifs = os.path.join(model_interpolation_folder, path + "_gifs")
    check_folder(model_path_gifs)

    categorie_path = os.path.join(categorie_interpolation_folder, path)
    check_folder(categorie_path)

    categorie_path_gifs = os.path.join(categorie_interpolation_folder, path + "_gifs")
    check_folder(categorie_path_gifs)

    return model_path, model_path_gifs, categorie_path, categorie_path_gifs


def create_gif(path, output_path):
    # ffmpeg -framerate 10 -i results/embed_n_to_n/image-%d.png results/test/output.gif
    os.system(f"ffmpeg -framerate 10 -i {path}image-%d.png {output_path} -v 0")

def gen_gifs_from_images(temp_folder, 
                         model_inter_folder, model_inter_folder_gifs,
                         categorie_inter_folder, categorie_inter_gifs, 
                         model_name, interpolation_title,
                         k,
                         images, titles):
    # run command line to create gif : 
    create_gif(temp_folder, os.path.join(model_inter_folder_gifs, f"{interpolation_title}{k}.gif"))

    shutil.copy(os.path.join(model_inter_folder_gifs, f"{interpolation_title}{k}.gif"),
                os.path.join(categorie_inter_gifs, model_name + f"{interpolation_title}{k}.gif"))

    image_generator.show_generated_images(
        images, 
        title="Interpolation between two images",
        titles=titles,
        path=os.path.join(model_inter_folder, f"{interpolation_title}{k}.png"))
    
    shutil.copy(os.path.join(model_inter_folder, f"{interpolation_title}{k}.png"),
                os.path.join(categorie_inter_folder, model_name + f"{interpolation_title}{k}.png"))
    
    # copy all image from temp folder to model folder, rename from image-0 to embed_n_to_n_k_0
    for filename in os.listdir(temp_folder):
        shutil.copy(os.path.join(temp_folder, filename), model_inter_folder)
        os.rename(os.path.join(model_inter_folder, filename), 
                  os.path.join(model_inter_folder, interpolation_title + str(k) + "_" + filename))

    # copy all image from temp folder to categories folder, rename from image-0 to model_name_embed_n_to_n__k_0
    for filename in os.listdir(temp_folder):
        shutil.copy(os.path.join(temp_folder, filename), categorie_inter_folder)
        os.rename(os.path.join(categorie_inter_folder, filename), 
                  os.path.join(categorie_inter_folder, model_name + interpolation_title + filename))

    # remove the temp folder
    shutil.rmtree(temp_folder)


def gen_embed_n_to_n(INTERPOLATION, 
                     autoencoder_model, model_name, 
                     model_interpolation_folder, categorie_interpolation_folder, 
                     image_generator, classes, dataset, ref_labels):
    interpolation_title = "embed_n_to_n"
    print("Creating interpolation " + interpolation_title + " ... \n")

    number_time = 5
    start_index = 0

    model_embed_n_to_n_folder, model_embed_n_to_n_folder_gifs, \
        categorie_embed_n_to_n_folder, categorie_embed_n_to_n_folder_gifs = create_folder(
            model_interpolation_folder, categorie_interpolation_folder, interpolation_title)

    for k in range(number_time):
        temp_folder = os.path.join(model_interpolation_folder, interpolation_title + "_temp/")
        check_folder(temp_folder, clear=True)

        images, image_labels, image_indices = DatasetUtils.get_n_first_for_label(
                    dataset, 
                    ref_labels[0], N_IMAGES, start_index=start_index)
        
        images.append(images[0])
        image_labels.append(image_labels[0])
        image_indices.append(image_indices[0])

        encoded = []
        titles = []

        for i in range(len(images)):
            if i < len(images) - 1:
                for _ in range(INTERPOLATION):
                    titles.append(f"{classes[image_labels[i]]} ({i}) to {classes[image_labels[i + 1]]} ({i + 1})")
            pack = autoencoder_model(PytorchUtils.numpy_to_tensor(images[i], device).unsqueeze(0), 
                                    torch.tensor(image_labels[i]).to(device).unsqueeze(0))
            encoded.append(pack['encoded'])

        images = image_generator.interpolate_vectors(
            encoded, 
            interpolation_number=INTERPOLATION, 
            save_path=temp_folder, 
            image_size=(None if autoencoder_model.latent_type == "convolutional" else (WIDTH, HEIGHT)))

        gen_gifs_from_images(temp_folder,
                                model_embed_n_to_n_folder, model_embed_n_to_n_folder_gifs,
                                categorie_embed_n_to_n_folder, categorie_embed_n_to_n_folder_gifs,
                                model_name, interpolation_title + "_", 
                                k,
                                images, titles)
        
        start_index = image_indices[N_IMAGES - 1] + 1

def gen_embed_n_to_m(INTERPOLATION, 
                     autoencoder_model, model_name, 
                     model_interpolation_folder, categorie_interpolation_folder,
                     image_generator, classes, dataset, ref_labels):
    interpolation_title = "embed_n_to_m"
    print("Creating interpolation " + interpolation_title + " ... \n")

    number_time = 5
    start_indices = [0] * len(ref_labels)

    model_embed_n_to_m_folder, model_embed_n_to_m_folder_gifs, \
        categorie_embed_n_to_m_folder, categorie_embed_n_to_m_folder_gifs = create_folder(
            model_interpolation_folder, categorie_interpolation_folder, interpolation_title)

    for k in range(number_time):
        temp_folder = os.path.join(model_interpolation_folder, interpolation_title + "_temp/")
        check_folder(temp_folder, clear=True)

        images, image_labels, image_indices = [], [], []

        add(DatasetUtils.get_n_first_for_label(dataset, ref_labels[0], 1, start_index=start_indices[0]), 
            images, image_labels, image_indices)
        add(DatasetUtils.get_n_first_for_label(dataset, ref_labels[1], 1, start_index=start_indices[1]), 
            images, image_labels, image_indices)
        add(DatasetUtils.get_n_first_for_label(dataset, ref_labels[2], 1, start_indices[2]), 
            images, image_labels, image_indices)
        
        images.append(images[0])
        image_labels.append(image_labels[0])
        image_indices.append(image_indices[0])

        encoded = []
        titles = []

        for i in range(len(images)):
            if i < len(images) - 1:
                for _ in range(INTERPOLATION):
                    titles.append(f"{classes[image_labels[i]]} ({i}) to {classes[image_labels[i + 1]]} ({i + 1})")

            pack = autoencoder_model(PytorchUtils.numpy_to_tensor(images[i], device).unsqueeze(0), 
                                     torch.tensor(image_labels[i]).to(device).unsqueeze(0))
            encoded.append(pack['encoded'])

        images = image_generator.interpolate_vectors(
            encoded, 
            interpolation_number=INTERPOLATION, 
            save_path=temp_folder, 
            image_size=(None if autoencoder_model.latent_type == "convolutional" else (WIDTH, HEIGHT)))

        gen_gifs_from_images(temp_folder,
                                model_embed_n_to_m_folder, model_embed_n_to_m_folder_gifs,
                                categorie_embed_n_to_m_folder, categorie_embed_n_to_m_folder_gifs,
                                model_name, interpolation_title+"_", 
                                k,
                                images, titles)
        
        for l in range(len(start_indices)):
            start_indices[l] = image_indices[l] + 1


def gen_n_to_m_embed_before_after(INTERPOLATION, 
                     autoencoder_model, model_name, 
                     model_interpolation_folder, categorie_interpolation_folder,
                     image_generator, classes, dataset, ref_labels, is_before):
    interpolation_title = "n_to_m_embed_" + ("before" if is_before else "after")

    print("Creating interpolation " + interpolation_title + " ... \n")

    number_time = 5

    start_indices = [0] * len(ref_labels)

    model_n_to_m_embed_before_folder, model_n_to_m_embed_before_folder_gifs, \
        categorie_n_to_m_embed_before_folder, categorie_n_to_m_embed_before_folder_gifs = create_folder(
            model_interpolation_folder, categorie_interpolation_folder, interpolation_title)
    
    for k in range(number_time):
        temp_folder = os.path.join(model_interpolation_folder, interpolation_title + "_temp/")
        check_folder(temp_folder, clear=True)

        images, image_labels, image_indices = [], [], []

        add(DatasetUtils.get_n_first_for_label(dataset, ref_labels[0], 1, start_index=start_indices[0]), 
            images, image_labels, image_indices)
        add(DatasetUtils.get_n_first_for_label(dataset, ref_labels[1], 1, start_index=start_indices[1]), 
            images, image_labels, image_indices)
        add(DatasetUtils.get_n_first_for_label(dataset, ref_labels[2], 1, start_indices[2]), 
            images, image_labels, image_indices)

        images.append(images[0])
        image_labels.append(image_labels[0])
        image_indices.append(image_indices[0])

        encoded = []
        titles = []

        for i in range(len(images)):
            if i < len(images) - 1:
                for j in range(INTERPOLATION):
                    titles.append(f"{classes[image_labels[i]]} ({i}) to {classes[image_labels[i + 1]]} ({i + 1})")

            pack = autoencoder_model(PytorchUtils.numpy_to_tensor(images[i], device).unsqueeze(0), 
                                     torch.tensor(image_labels[i]).to(device).unsqueeze(0))
            encoded.append(pack['encoded'])

        if not is_before:
            # move labels by one
            del image_labels[0]

        images = image_generator.interpolate_vectors(
            encoded,
            interpolation_number=INTERPOLATION,
            labels=image_labels,
            save_path=temp_folder,
            image_size=(None if autoencoder_model.latent_type == "convolutional" else (WIDTH, HEIGHT)))
        
        gen_gifs_from_images(temp_folder,
                                model_n_to_m_embed_before_folder, model_n_to_m_embed_before_folder_gifs,
                                categorie_n_to_m_embed_before_folder, categorie_n_to_m_embed_before_folder_gifs,
                                model_name, interpolation_title+"_", 
                                k,
                                images, titles)
        
        for l in range(len(start_indices)):
            start_indices[l] = image_indices[l] + 1

def gen_n_to_n_embed_other(INTERPOLATION, 
                     autoencoder_model, model_name, 
                     model_interpolation_folder, categorie_interpolation_folder,
                     image_generator, classes, dataset, ref_labels):
    interpolation_title = "n_to_n_embed_other"
    print("Creating interpolation " + interpolation_title + " ... \n")

    start_index = 0

    model_embed_n_to_n_folder, model_embed_n_to_n_folder_gifs, \
        categorie_embed_n_to_n_folder, categorie_embed_n_to_n_folder_gifs = create_folder(
            model_interpolation_folder, categorie_interpolation_folder, interpolation_title)

    for k in range(len(classes)):
        temp_folder = os.path.join(model_interpolation_folder, interpolation_title + "_temp/")
        check_folder(temp_folder, clear=True)

        images, image_labels, image_indices = DatasetUtils.get_n_first_for_label(
                    dataset, 
                    ref_labels[0], N_IMAGES, start_index=start_index)
        
        images.append(images[0])
        image_labels.append(image_labels[0])
        image_indices.append(image_indices[0])

        encoded = []
        titles = []

        for i in range(len(images)):
            if i < len(images) - 1:
                for _ in range(INTERPOLATION):
                    titles.append(f"{classes[image_labels[i]]} ({i}) to {classes[image_labels[i + 1]]} ({i + 1})")
            pack = autoencoder_model(PytorchUtils.numpy_to_tensor(images[i], device).unsqueeze(0), 
                                    torch.tensor(image_labels[i]).to(device).unsqueeze(0))
            encoded.append(pack['encoded'])

        images = image_generator.interpolate_vectors(
            encoded, 
            interpolation_number=INTERPOLATION, 
            save_path=temp_folder, 
            image_size=(None if autoencoder_model.latent_type == "convolutional" else (WIDTH, HEIGHT)))

        gen_gifs_from_images(temp_folder,
                                model_embed_n_to_n_folder, model_embed_n_to_n_folder_gifs,
                                categorie_embed_n_to_n_folder, categorie_embed_n_to_n_folder_gifs,
                                model_name, interpolation_title + "_", 
                                k,
                                images, titles)

        # keep at start index 0, because we want to test the addition of embeddings on same images

if __name__ == "__main__":
    with torch.no_grad():
        parser = create_arg_parse()
        args = parser.parse_args()

        if args.replace:
            print("Replace file path by example model file")
            with open(os.path.join("resources", args.file), "w") as f:
                json.dump(return_example_model_file(), f, indent=4)
            exit(0)

        if not args.file:
            print("No file given, using default : models.json")
            args.file = "models.json"

        with open(os.path.join("resources", args.file), "r") as f:
            models = json.load(f)

        previous_dim = [None, None]

        train_dataset = None
        training_loaded_set = None
        test_dataset = None
        test_loaded_set = None


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
        training_data, training_labels, validation_data, validation_labels, test_data, test_labels = data_manager.parse_data(data)

        LEN_SUBSET = 100
        TEST_LEN_SUBSET = 20
        STROKE_SIZES = {
            128 : 3,
            64 : 2
        }

        # Take each part of the dataset for each class of size LEN_SUBSET
        training_data_subset = np.array([])
        training_labels_subset = np.array([], dtype=int)

        testing_data_subset = np.array([])
        testing_labels_subset = np.array([], dtype=int)

        sum_train = 0
        sum_test = 0
        for i in range(nb_classes):

            training_data_subset = np.append(training_data_subset, training_data[sum_train:sum_train + LEN_SUBSET])
            training_labels_subset = np.append(training_labels_subset, training_labels[sum_train:sum_train + LEN_SUBSET])

            testing_data_subset = np.append(testing_data_subset, test_data[sum_test:sum_test + TEST_LEN_SUBSET])
            testing_labels_subset = np.append(testing_labels_subset, test_labels[sum_test:sum_test + TEST_LEN_SUBSET])

            sum_train += len(data[i]["train"])
            sum_test += len(data[i]["test"])

        IMAGE_FACTOR = 1.1

        print(len(training_data_subset), len(training_labels_subset), len(testing_data_subset), len(testing_labels_subset))
        print(type(training_data), type(training_data_subset))

        output_folder = os.path.join("results", "images")
        check_folder(output_folder)

        # Check model and categories folder
        model_folder = os.path.join(output_folder, "model")
        check_folder(model_folder)
        
        categories_folder = os.path.join(output_folder, "categories")
        check_folder(categories_folder)

        for model in models["models"]:
            print(f"Model : {model['model']}, path : {model['path']}")

            parameters = model["parameters"]

            WIDTH = parameters["image_size"][0]
            HEIGHT = parameters["image_size"][1]

            CONV_MODEL_DATA = "conv" in model["model"] or "convolutionnal" in model["model"]
            
            # If image size changes, or if it's the first iteration 
            # dataset must be initialized
            if previous_dim[0] != WIDTH or previous_dim[1] != HEIGHT:
                print("Reload dataset for images  : " + str(parameters["image_size"]))
                train_dataset = StrokeImageDataset(
                    data=training_data_subset, 
                    labels=training_labels_subset,
                    width=WIDTH, height=HEIGHT,
                    stroke_size=STROKE_SIZES[parameters["image_size"][0]], factor=IMAGE_FACTOR, 
                    reshape=CONV_MODEL_DATA, normalize=True)
                
                test_dataset = StrokeImageDataset(
                    data=testing_data_subset, 
                    labels=testing_labels_subset,
                    width=WIDTH, height=HEIGHT,
                    stroke_size=STROKE_SIZES[parameters["image_size"][0]], factor=IMAGE_FACTOR, 
                    reshape=CONV_MODEL_DATA, normalize=True)
                
                training_loaded_set = DataLoader(train_dataset, batch_size=parameters["batch_size"], shuffle=False)
                test_loaded_set = DataLoader(test_dataset, batch_size=parameters["batch_size"], shuffle=False)

                print(f'training set distribution :\n{Visualisation.dataset_distribution(classes, train_dataset)}')
                print(f'test set distribution :\n{Visualisation.dataset_distribution(classes, test_dataset)}')

            CONV_ARCHITECTURE = [1, 32, 16, 8]
            LINEAR_ARCHITECTURE = [WIDTH * HEIGHT, WIDTH * HEIGHT * 3 // 4]

            """
            INIT DEVICE
            """
            force_cpu = "force_cpu" in parameters.keys() 
            if force_cpu:
                force_cpu = parameters["force_cpu"]

            device = PytorchUtils.device_section(force_cpu=force_cpu) 
            print(f"Device : {device}")

            autoencoder_model, is_embed_model = InitModel.init_model(
                model["model"], 
                device, 
                WIDTH, HEIGHT, classes, 
                parameters["conv_architecture"] if "conv_architecture" in parameters.keys()  else CONV_ARCHITECTURE,
                parameters["linear_architecture"] if "linear_architecture" in parameters.keys()  else LINEAR_ARCHITECTURE, 
                parameters["dropout"], parameters["batch_norm"], 
                parameters["rl"] if "rl" in parameters.keys() else 1, 
                parameters["kl"] if "kl" in parameters.keys() else 0,
                verbose=False)
            
            InitModel.print_model_characteristics(autoencoder_model)
            
            criterion, optimizer = InitModel.create_criterion_optimizer(model["model"], autoencoder_model, parameters["lr"])

            path = os.path.join("weights", model["path"] + ".pt")

            # Load weights
            autoencoder_model.train_autoencoder(train_dataset, train_dataset, optimizer, criterion, parameters["epochs"], path=path)

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

            model_name = model["model"] + "_" + str(WIDTH) + "x" + str(HEIGHT)
            if "rl" in parameters.keys():
                model_name += "_rl_" + str(parameters["rl"]) + "_kl_" + str(parameters["kl"])

            print("Creating images for model : " + model_name)
            
            model_output_folder = os.path.join(model_folder, model_name)
            # Check if exitst
            check_folder(model_output_folder)

            do = {
                "train_curve" : True,
                "reconstruction" : True,
                "latent_space" : True,
                "mean_vectors" : True,
                "modified_mean_vectors" : True,
                "embed_images" : True,
                "interpolation_embed_n_to_n" : True,
                "interpolation_embed_n_to_m" : True,
                "interpolation_n_to_m_embed_before" : True,
                "interpolation_n_to_m_embed_after" : True,
                "interpolation_n_to_n_embed_other" : True
            }

            """
            TRAIN CURVE
            """

            if do["train_curve"]:
                gen_train_curve(autoencoder_model, model_output_folder, categories_folder, model_name)

            """
            RECONSTRUCTION EXAMPLES
            """
            if do["reconstruction"]:
                gen_reconstruction(autoencoder_model, train_dataset, 
                               test_dataset, model_output_folder, categories_folder, model_name)

            """
            LATENT SPACES
            """
            if do["latent_space"]:
                gen_latent_space(autoencoder_model, training_loaded_set, 
                             model_output_folder, categories_folder, model_name, is_embed_model)
                
            """
            MEAN VECTORS
            """
            if do["mean_vectors"]:
                mean_encoded_information = gen_mean_vectors(autoencoder_model, training_loaded_set, 
                                                        model_output_folder, categories_folder, model_name, image_generator)

            """
            GENERATED MODIFIED MEAN VECTORS
            """
            if do["modified_mean_vectors"]: 
                gen_modified_mean_vectors(autoencoder_model,
                                        model_output_folder, categories_folder, model_name, image_generator, classes)

            if is_embed_model and do["embed_images"]:
                """
                EMBED IMAGES WITH OTHER CLASSES
                """
                gen_embed_images_with_other_classes(autoencoder_model,
                                                    model_output_folder, categories_folder, model_name, 
                                                    image_generator, classes, train_dataset)
                        
            """ INTERPOLATION PART """

            INTERPOLATION = 8
            N_IMAGES = 3

            def add(tuple, a, b, c):
                a += tuple[0]
                b += tuple[1]
                c += tuple[2]

            ref_labels = [0, 1, 2]

            model_interpolation_folder = os.path.join(model_output_folder, "interpolation")
            check_folder(model_interpolation_folder)

            categorie_interpolation_folder = os.path.join(categories_folder, "interpolation")
            check_folder(categorie_interpolation_folder)


            """
            INTERPOLATION embed_n_to_n
            """

            if do["interpolation_embed_n_to_n"]:
                gen_embed_n_to_n(INTERPOLATION,
                                autoencoder_model, model_name,
                                model_interpolation_folder, categorie_interpolation_folder,
                                image_generator, classes, train_dataset, ref_labels)
            """
            INTERPOLATION embed_n_to_m
            """

            if do["interpolation_embed_n_to_m"]:
                gen_embed_n_to_m(INTERPOLATION,
                                autoencoder_model, model_name,
                                model_interpolation_folder, categorie_interpolation_folder,
                                image_generator, classes, train_dataset, ref_labels)

            """
            INTERPOLATION n_to_m_embed_before
            """

            if is_embed_model and do["interpolation_n_to_m_embed_before"]:
                gen_n_to_m_embed_before_after(INTERPOLATION,
                                            autoencoder_model, model_name,
                                            model_interpolation_folder, categorie_interpolation_folder,
                                            image_generator, classes, train_dataset, ref_labels, True)

            """
            INTERPOLATION n_to_m_embed_after
            """

            if is_embed_model and do["interpolation_n_to_m_embed_after"]:
                gen_n_to_m_embed_before_after(INTERPOLATION,
                                            autoencoder_model, model_name,
                                            model_interpolation_folder, categorie_interpolation_folder,
                                            image_generator, classes, train_dataset, ref_labels, False)

            """
            INTERPOLATION n_to_n_embed_other
            """

            if is_embed_model and do["interpolation_n_to_n_embed_other"]:
                gen_n_to_n_embed_other(INTERPOLATION,
                                        autoencoder_model, model_name,
                                        model_interpolation_folder, categorie_interpolation_folder,
                                        image_generator, classes, train_dataset, ref_labels)    
                
            plt.close('all')