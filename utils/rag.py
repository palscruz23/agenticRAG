from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import openai
from dotenv import find_dotenv, load_dotenv
import streamlit as st
import os

# Load environment variables
load_dotenv(find_dotenv())

from utils.get_embeddings import get_embedding_function
import sys

api_key = os.getenv("OPENAI_API_KEY")

CHROMA_PATH = "./utils/chroma"

PROMPT_TEMPLATE = """

CONTEXT INFORMATION:
{context}

USER QUERY:
"{query}"

INSTRUCTIONS:
- Answer the query using ONLY the information provided in the context above
- If the context doesn't contain enough information to answer the query, clearly state this
- Cite specific parts of the context when making claims
- Be concise but complete in your response
- If multiple interpretations are possible, acknowledge this

ANSWER:
"""

def query_rag(query: str):
    ### QA type
    # Prepare the DB.
    embedding_function = get_embedding_function()
    # db = Chroma(
    #     persist_directory=CHROMA_PATH,
    #     embedding_function=get_embedding_function(),
    # )
    st.session_state.vectorstore = Chroma(
        embedding_function=get_embedding_function()
    )
    # Search the DB.
    query_context = query
    results = st.session_state.vectorstore.similarity_search_with_score(query_context, k=5)
    results = list(reversed(results))

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, query=query)

    model = ChatOpenAI(model_name= "gpt-4o", api_key=api_key)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text.content}"
    return formatted_response, sources

def context_rag(query: str):
    ### QA type
    # Prepare the DB.
    embedding_function = get_embedding_function()
    st.session_state.vectorstore = Chroma(
        embedding_function=get_embedding_function()
    )

    # Search the DB.
    query_context = query
    results = st.session_state.vectorstore.similarity_search_with_score(query_context, k=5)
    results = list(reversed(results))

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, query=query)

    model = ChatOpenAI(model_name= "gpt-4o", api_key=api_key)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    content = [doc.page_content for doc, __ in results]
    return results, sources