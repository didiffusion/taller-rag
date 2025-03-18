'''
Input Data:
A set of short text documents with information about AI technologies.
A set of sample questions related to these documents.

Requirements:
Simple Retrieval: Implement a basic retrieval mechanism to find the most relevant document based on the query.
Simple Answer Generation: Use an open-source text generation model to generate a response based on the retrieved document.
Integration: Create a simple flow that combines retrieval and generation to answer questions.

Instructions:
Setup: Load and preprocess the documents.
Retrieval: Implement a basic retrieval method (e.g., TF-IDF).
Generation: Use a pre-trained, open-source LLM to generate answers based on the retrieved document.
Documentation: Provide a brief explanation of how you implemented retrieval and generation.

Evaluation Questions:
How did you implement the retrieval mechanism and why?
How did you choose the LLM and what are its limitations?
How could this solution be improved or scaled?
'''


'''
1. Read docs
2. Get doc embeddings
3. Store on memory
4. Prompt the user
5. Retrieve most similar document on db
6. Put it in context
7. Answer the user
'''

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

from ollama import chat
from ollama import ChatResponse

documents = [
    "Artificial intelligence is transforming the medical field with new diagnostic and treatment options.",
    "Machine learning is increasingly used for analyzing large data sets in finance.",
    "Recent advancements in natural language processing include better context understanding."
]
vector_db = None
vectorizer = TfidfVectorizer()

def get_vectors_from_documents(docs):
    vectors = vectorizer.fit_transform(docs)
    return vectors

def get_tf_idf_query_similarity(vectorizer, docs_tfidf, query):
    query_tfidf = vectorizer.transform([query])
    cosineSimilarities = cosine_similarity(query_tfidf, docs_tfidf).flatten()
    return cosineSimilarities

def main():
    vector_db = get_vectors_from_documents(documents)
    user_prompt = input("Enter question:")
    user_vector = vectorizer.fit_transform([user_prompt])
    docs_cosine_similarity = get_tf_idf_query_similarity(vectorizer,get_vectors_from_documents(documents),user_prompt)
    relevant_doc_index = np.argmax(docs_cosine_similarity)
    query = documents[relevant_doc_index]
    call_ollama(user_prompt, query)


def call_ollama(user_prompt, query):
    response: ChatResponse = chat(model='llama3', messages=[
        {
            'role': 'system',
            'content': 'You are a helpful AI assistant. Answer the user\'s question based on the provided context. If the context is not relevant to the question, say so politely.'
        },
        {
            'role': 'user',
            'content': f'Question: {user_prompt}\n\nContext: {query}\n\nPlease answer the question based on this context.'
        }
    ])
    print("\nAnswer:")
    print(response.message.content)


main()
