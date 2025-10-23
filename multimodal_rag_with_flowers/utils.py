from matplotlib import pyplot as plt
from PIL import Image
import os

def display_image(img):
    """Display an image using matplotlib."""
    plt.imshow(img)
    plt.axis("off")
    plt.show()

def show_image_from_uri(uri):
    """Display an image from a given URI."""
    img = Image.open(uri)
    display_image(img)

def save_images(dataset, dataset_folder, num_images):
    """Save images from the dataset to the specified folder."""
    for i in range(num_images):
        if (i + 1) % 20 == 0:
            print(f"  Saving image {i+1} of {num_images}")
        image = dataset["train"][i]["image"]
        image.save(os.path.join(dataset_folder, f"flower_{i+1}.png"))
    print(f"Saved the first {num_images} images to {dataset_folder}")
