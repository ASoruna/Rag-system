# Rag-system

Overview
This Python script demonstrates the creation and use of a text retrieval system leveraging advanced natural language processing (NLP) techniques. It integrates a HuggingFace embedding model, a large language model (LLM) from Llama, and a PostgreSQL vector database to store, process, and query text documents efficiently. The main purpose of this script is to allow for the querying of documents using natural language, retrieving relevant information based on vector similarity.

Features
Document Loading: The script loads PDF documents using the PyMuPDFReader.
Sentence Splitting: Loaded documents are split into smaller chunks (sentences) using the SentenceSplitter for more granular processing.
Vector Embeddings: The script uses the HuggingFace model to generate embeddings for each text chunk.
Vector Storage: Embeddings are stored in a PostgreSQL vector database (PGVectorStore).
Custom Retriever: A custom retriever (VectorDBRetriever) is implemented to query the database and retrieve relevant text chunks based on a query.
LLM Integration: The script uses a Llama model to generate more coherent responses by processing the retrieved information.
Setup Instructions
Prerequisites
Python 3.x
PostgreSQL installed and running locally.
Necessary Python libraries:
llama_index
psycopg2
sqlalchemy
PyMuPDF
Installation
Clone the Repository

git clone https://github.com/ASoruna/Rag-system.git
cd your-repo
Install Dependencies

Install the required Python libraries:

pip install llama_index psycopg2 sqlalchemy PyMuPDF
Configure PostgreSQL

Ensure PostgreSQL is installed and running locally. You can modify the database credentials in the script as needed:

python
db_name = "vector_db"
host = "localhost"
password = "password"  # postgresql password
port = "5432"
user = "postgres"
Download the PDF Document

Place your PDF document in the ./data/ directory. For example, SagiriusJrAndrewEssay3.pdf.

Running the Script
Run the script using Python:

python script_name.py
Usage

Loading and Splitting Documents: The script loads the provided PDF document, splits it into sentences, and stores them in the database after embedding.
Querying the Database: You can query the database using natural language queries. The script retrieves relevant text chunks, which are processed by the LLM for a final response.
Example query:

python
query_str = "What is Vladimir's role in the late game?"
response = query_engine.query(query_str)
The script will print both the generated response and the context from the source document.

Customization
Embedding Model: The embedding model can be swapped out for any HuggingFace model by changing the model_name parameter.
LLM Configuration: The Llama model's settings can be adjusted, such as the temperature and the number of tokens generated.
Database Connection: Modify the PostgreSQL connection parameters to connect to a different database.


Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.

Acknowledgements
This project uses the following open-source projects:

LlamaCPP
HuggingFace
psycopg2
PyMuPDF
