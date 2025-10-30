# Multimodal Florist
This project will (eventually) contain a shiny app features an LLM-powered chatbot to assist with flower arrangements. The chatbot will be making use of a multimodal RAG store containing images for flowers.

## Details
- `db_setup`
  - Loads the Flowers 102 dataset using Hugging Face Datasets library.
  - Saves the first 500 images as PNGs to a local directory AND to a vector db (chroma).
- `test.py`
  - Test use of the multimodal RAG vector store.
- `app.py` (Unfinished) 
  - Shiny app utilizing the above.
