# Product Pricer
LLMs are used in a *regression* task to estimate the prices of products based on their descriptions.

## Details
This project currently consists of 3 notebooks as follows:
- [Part 1](llm_regression.ipynb)
  - Frontier LLMs are used in a regression task to estimate the prices of products based on their descriptions.
  - Baseline: Random Forest, Support Vector Machines along with Bag-of-Words and Word2Vec features.
  - Frontier LLMs used: ChatGPT-4o, ChatGPT-4o-mini, Claude 3.5 Sonnet.
- [Part 2](llm_regression_2.ipynb)
  - An open-source LLM (Llama-3.1-8B) is specifically fine-tuned to predict prices of Amazon products based on their descriptions with *LoRA (Low Rank Adaptation)*.
  - Model being trained is pushed to Hugging Face every 2000 steps.
  - Performance on a validation set is monitored as the model is being trained (Weights & Biases).
  - *Model quantization* to reduce memory footprint.
  - *Data collation* to guide the model on what to focus on learning how to predict.
- [Part 3](llm_regression_3.ipynb)
  - The fine-tuned model from [part 2](llm_regression/llm_regression_2.ipynb) is used to predict prices of Amazon products in the test set used previously in [part 1](llm_regression/llm_regression.ipynb).
