from utils import *
import base64

# pip install langchain-community langchain-core langchain-openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# === For Testing ONlY === uncomment to test
query = "purple petals"  # Change the query to test different images or different
results = query_db(query)
print_results(results)

# =================================
# === Setting up the RAG Flow ===

# 1. the user submits a query (question, query, etc)
# 2. the query is sent to the multimodal database (retrieval function first)
## * in our case, we try to pull the images that match the user's query
# 3. Those images are then passed (along with prompt) to the a vision model where it will use the images context and respond to the prompt as a final output


# Instantiate the OpenAI model
vision_model = ChatOpenAI(
    model="gpt-4o", temperature=0.0
)  # this model has vision capabilities

# instantiate the output parser
parser = StrOutputParser()


# Define the prompt template
image_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a talented florist and you have been asked to create a bouquet of flowers for a special event. Answer the user's question  using the given image context with direct references to parts of the images provided."
            " Maintain a more conversational tone, don't make too many lists. Use markdown formatting for highlights, emphasis, and structure.",
        ),
        (
            "user",
            [
                {
                    "type": "text",
                    "text": "what are some good ideas for a bouquet arrangement {user_query}",
                },
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64,{image_data_1}",
                },
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64,{image_data_2}",
                },
            ],
        ),
    ]
)

# Define the LangChain Chain
vision_chain = image_prompt | vision_model | parser


# === Foramtting query results for LLM prompting ===
# to input the images in as context, we need to first encode the images as base64 strings for the LLM to understand


# The function below that will do that, and create a dictionary along with
# the original user query to pass into the chain. The chain will take a dictionary input,
# that will correspond to the three pieces of information
# that need to be injected into it {user_query}, {image_data_1}, {image_data_2}.
def format_prompt_inputs(data, user_query):
    print("Formatting prompt inputs...")
    inputs = {}

    # Add user query to the dictionary
    inputs["user_query"] = user_query

    # Get the first two image paths from the 'uris' list
    # print('len(data):', len(data))
    # print('len(data["uris"]):', len(data["uris"]))
    # print('len(data["uris"][0]):', len(data["uris"][0]))
    # print(data)
    # print('===============================')
    # print('data["uris"][0]:', data["uris"][0])
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


## === Putting it all together ===
print("Welcome to the flower arrangement service!")
print("Please enter your query to get some ideas for a bouquet arrangement.")

query = input("Enter your query: \n")

# Running Retrieval and Generation
results = query_db(query, results=2)
prompt_input = format_prompt_inputs(results, query)
response = vision_chain.invoke(prompt_input)

print("\n ------- \n")

print("\n ---Response---- \n")
print(response)


# Display the retrieved images
print("\n Here are some ideas for a bouquet arrangement based on your query: \n")
show_image_from_uri(results["uris"][0][0])
show_image_from_uri(results["uris"][0][1])

print("\n Images URI: \n")  
print(f"Image 1: {results['uris'][0][0]}")
print(f"Image 2: {results['uris'][0][1]}")
