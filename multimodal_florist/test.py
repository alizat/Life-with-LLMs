from utils import *

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

# pip install langchain-community langchain-core langchain-openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

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

# # preliminary test to make sure it works
# print('\n---------------------\n')
# print('Preliminary Test...\n')
# query = "purple petals"  # Change the query to test different images or different
# results = query_db(flower_collection, query)
# print_results(results)
# print('\n---------------------\n')

# RAG Flow
# 1. the user submits a query
# 2. the query is sent to the multimodal database to retrieve images that match the user's query
# 3. Those images are then passed (along with prompt) to the a vision model where it will use the images context and respond to the prompt as a final output.

# Instantiate the OpenAI model
vision_model = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=OPENAI_API_KEY)  # this model has vision capabilities

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

# The Actual Test
print("Welcome to the flower arrangement service!")
print("Please enter your query to get some ideas for a bouquet arrangement.")
query = input("Enter your query: \n")  # yellow flowers

# Running Retrieval and Generation
results      = query_db(flower_collection, query, results=2)  # get two most relevant images to the query
prompt_input = format_prompt_inputs(results, query)  # format the prompt input for the LLM (query + 2 relevant images from RAG store)
response     = vision_chain.invoke(prompt_input)  # get LLM response

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
