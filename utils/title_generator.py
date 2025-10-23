from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import openai
from dotenv import find_dotenv, load_dotenv
import streamlit as st
import os

# Load environment variables
load_dotenv(find_dotenv())

import sys

api_key = os.getenv("OPENAI_API_KEY")

PROMPT_TEMPLATE = """
You are given a list of document titles retrieved from a knowledge base and a user query. Your task is to generate a concise title description for the chatbot. Also, generate a generic topic aligned from the title.

DOCUMENT TITLES:
{context}

USER QUERY:
{query}

INSTRUCTIONS:
- Limit the title description to a maximum of 5 words.
- Limit the title topic to a maximum of 5 words.
- Follow the answer format - title, topic

ANSWER:
title, topic
"""

def title_llm(context_text: str):
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, query="What is the best title and topic for a chatbot based on the following documents?")

    model = ChatOpenAI(model_name= "gpt-4o", api_key=api_key)
    response_text = model.invoke(prompt)
    response = tuple(response_text.content.split(", "))
    try:
        return response[0], response[1]
    except IndexError:
        print(f"✗ IndexError: Index is out of range for tuple with length {len(response)}")
        return "Personal Chatbot", "anything in the PDF files"
