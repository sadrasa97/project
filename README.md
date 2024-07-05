
# PDF CHATBOT

## Overview

This project integrates a question-answering system with evaluation metrics using BERT embeddings for scoring. The primary components include:
1. Retrieval Augmentation using `IndoxRetrievalAugmentation`.
2. Question Answering with `mistralai/Mistral-7B-Instruct-v0.2`.
3. Embedding with `multi-qa-mpnet-base-cos-v1`.
4. Evaluation using BERTScore.

## Requirements

- Python 3.6 or higher
- Required Python packages:
  - pandas
  - numpy
  - scikit-learn
  - torch
  - transformers
  - indox

## Setup

### Install Dependencies

```bash
pip install pandas numpy scikit-learn torch transformers indox
```

### Download Pre-trained Models

The project uses the following pre-trained models:
- BERT base uncased: `bert-base-uncased`
- Mistral QA model: `mistralai/Mistral-7B-Instruct-v0.2`
- Embedding model: `multi-qa-mpnet-base-cos-v1`

The models will be automatically downloaded when running the code for the first time.

## Usage

### Uploading Data

Use the `upload_file` function to upload the dataset. The file will be split and stored in a vector store for retrieval.

```python
file_path = upload_file()
```

### Loading and Splitting Data

Load and split the dataset into chunks.

```python
simpleLoadAndSplit = SimpleLoadAndSplit(file_path=file_path, remove_sword=False, max_chunk_size=200)
docs = simpleLoadAndSplit.load_and_chunk()
```

### Storing Data in Vector Store

Store the chunks in the vector store.

```python
db = ChromaVectorStore(collection_name="sample", embedding=embed)
indox.connect_to_vectorstore(vectorstore_database=db)
indox.store_in_vectorstore(docs)
```

### Question Answering

Interactively ask questions and get answers.

```python
while True:
    query = input("Enter your question (or 'exit' to stop): ")
    if query.lower() == 'exit':
        break

    retriever = indox.QuestionAnswer(vector_database=db, llm=mistral_qa, top_k=5)
    answer = retriever.invoke(query=query)
    context = retriever.context

    print("Answer: ", answer)

    inputs = {
        "question": query,
        "answer": answer,
        "context": context
    }

    result = evaluator(inputs)
    print(result)
```

### Evaluation

Evaluate the answers using BERTScore.

```python
evaluator = Evaluation()

inputs = {
    "question": query,
    "answer": answer,
    "context": context
}

result = evaluator(inputs)
print(result)
```

## Example

Here’s an example to demonstrate the usage of the script:

1. Upload a file containing the data.
2. Load and split the data into chunks.
3. Store the data in a vector store.
4. Ask a question and get an answer along with the evaluation scores.

```python
file_path = upload_file()
simpleLoadAndSplit = SimpleLoadAndSplit(file_path=file_path, remove_sword=False, max_chunk_size=200)
docs = simpleLoadAndSplit.load_and_chunk()

db = ChromaVectorStore(collection_name="sample", embedding=embed)
indox.connect_to_vectorstore(vectorstore_database=db)
indox.store_in_vectorstore(docs)

evaluator = Evaluation()

while True:
    query = input("Enter your question (or 'exit' to stop): ")
    if query.lower() == 'exit':
        break

    retriever = indox.QuestionAnswer(vector_database=db, llm=mistral_qa, top_k=5)
    answer = retriever.invoke(query=query)
    context = retriever.context

    print("Answer: ", answer)

    inputs = {
        "question": query,
        "answer": answer,
        "context": context
    }

    result = evaluator(inputs)
    print(result)
```
