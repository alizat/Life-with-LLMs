"""Setup script for loading the Flowers 102 dataset using Hugging Face Datasets library.
Saves the first 500 images as PNGs to a local directory AND to a vector db (chroma) if not already saved before."""

import warnings
warnings.filterwarnings("ignore")

import os
from utils import display_image, show_image_from_uri, save_images
from datasets import load_dataset

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader


# load the dataset
ds = load_dataset("huggan/flowers-102-categories")
print("\nLoaded the Flowers 102 dataset from Hugging Face Datasets library.")
print("  Number of images:", ds.num_rows)  # show number or rows

# # observe a random image from the dataset
# flower = ds["train"][78]["image"]
# display_image(flower)

# save images to local directory
dataset_folder = "./dataset/flowers-102-categories"  # folder to save images
os.makedirs(dataset_folder, exist_ok=True)           # create directory if not exists
if len(os.listdir(dataset_folder)) == 0:             # if no images, save them
    print(f"\nSaving images (PNGs) to local directory: '{dataset_folder}' ...")
    save_images(ds, dataset_folder, num_images=500)  # save the first 500 images
else:
    print(f"\nImages (PNGs) already saved in '{dataset_folder}'")

# # observe a random image from those saved in 'dataset_folder'
# random_flower = 'dataset/flowers-102-categories/flower_261.png'
# show_image_from_uri(random_flower)

# setup the chromaDB
chroma_client = chromadb.PersistentClient(path="./data/flower.db")  # create a chromadb object
image_loader = ImageLoader()                                        # instantiate image loader
embedding_function = OpenCLIPEmbeddingFunction()                    # instantiate multimodal embedding function

# create the collection (vector database)
flower_collection = chroma_client.get_or_create_collection(
    "flowers_collection",
    embedding_function=embedding_function,
    data_loader=image_loader,
)

# add images to vector db if not already added
if flower_collection.count() > 0:
    print(f"\nImages already exist in the database. Number of flowers saved: {flower_collection.count()}\n")
else:
    # get ids and uris of saved images
    ids = []
    uris = []
    for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
        if filename.endswith(".png"):
            file_path = os.path.join(dataset_folder, filename)
            ids.append(str(i))
            uris.append(file_path)

    # add images to the collection using add() method
    flower_collection.add(ids=ids, uris=uris)
    print(f"\nImages added to the database. Number of flowers saved: {flower_collection.count()}\n")
