from matplotlib import pyplot as plt
from PIL import Image
import os
import base64


# === Functions for Displaying/Saving Images ===
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


# === Functions for Querying the VectorDB and Printing Retrieval Results ===
def query_db(chroma_collection, query, results=5):
    """Query chroma collection and retrieve N most relevant items"""
    print(f"Querying the database for: {query}")
    results = chroma_collection.query(
        query_texts=[query], n_results=results, include=["uris", "distances"]
    )
    return results

def print_results(results):
    """Print results of query retrieval from chroma collection"""
    for idx, uri in enumerate(results["uris"][0]):
        print(f"\nID: {results['ids'][0][idx]}")
        print(f"Distance: {results['distances'][0][idx]}")
        print(f"Path: {uri}")
        # Display the image using matplotlib
        show_image_from_uri(uri)
        # print("\n")

# === Foramtting Query Results for LLM prompting ===
def format_prompt_inputs(data, user_query):
    """Encode first 2 images in data as base64 strings so they may later be passed as context to LLMs, 
    then return a dictionary containing the user query and the contents of the two images"""
    print("Formatting prompt inputs...")
    inputs = {}

    # Add user query to the dictionary
    inputs["user_query"] = user_query

    # Get the first two image paths from the 'uris' list
    image_path_1 = data["uris"][0][0]
    image_path_2 = data["uris"][0][1]

    # Encode the first image
    with open(image_path_1, "rb") as image_file:
        image_data_1 = image_file.read()
    inputs["image_data_1"] = base64.b64encode(image_data_1).decode("utf-8")

    # Encode the second image
    with open(image_path_2, "rb") as image_file:
        image_data_2 = image_file.read()
    inputs["image_data_2"] = base64.b64encode(image_data_2).decode("utf-8")

    # inputs dictionary will have the user query and the base64 encoded images and will look like this:
    # {
    #     "user_query": "pink flower with yellow center",
    #     "image_data_1": "base64_encoded_image_1",
    #     "image_data_2": "base64_encoded_image_2"
    # }
    print("Prompt inputs formatted....")
    return inputs
