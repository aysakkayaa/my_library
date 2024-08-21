# My Library

This project is a text classification system that uses `SentenceTransformer` for text embedding and `Qdrant` for vector similarity search. The system allows you to train a model with a dataset, store the embeddings in Qdrant, and perform similarity searches on test data.

## Features

- **Text Preprocessing**: Converts text to lower case, removes punctuation, and filters out stopwords.
- **Model Training**: Trains a `SentenceTransformer` model on a provided dataset.
- **Vector Storage**: Stores text embeddings in Qdrant for similarity search.
- **Similarity Search**: Finds the most similar texts to a given input using cosine similarity.

## Requirements

- Python 3.8+
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- [Docker](https://docs.docker.com/get-docker/) (for Qdrant)

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/aysakkayaa/my_library.git
cd my_library



### 2. Create a Conda Environment

Create and activate a new conda environment:

```bash

conda create -n my_library 
conda activate my_library

### 3. Install Dependencies

Install the required Python packages:

```bash

pip install -r requirements.txt

### 4. Run Qdrant with Docker

To store and search vectors, you need to run Qdrant. If you have Docker installed, you can run Qdrant using the following command:

```bash

docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

This command will pull the Qdrant image from Docker Hub (if not already downloaded) and run it on port 6333.

