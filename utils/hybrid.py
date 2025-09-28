from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import openai
from dotenv import find_dotenv, load_dotenv
import streamlit as st
import os

from utils.rag import context_rag, PROMPT_TEMPLATE
from utils.bm25 import context_bm25

# Load environment variables
load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")

def reciprocal_rank_fusion(rankings, k=60):
    """
    rankings: list of ranked lists from different retrievers
              e.g. [["doc1","doc2","doc3"], ["doc3","doc1","doc4"]]
    k: smoothing parameter
    """
    scores = {}
    for ranklist in rankings:
        for rank, doc in enumerate(ranklist, start=1):
            scores[doc] = scores.get(doc, 0) + 1.0 / (k + rank)

    # sort by highest RRF score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def hybrid_rag_bm25(query: str, tools: list[str] = ['rag', 'bm25'], hist_prompt: str = ""):

    rag_content, rag_sources = context_rag(query)
    bm25_content, bm25_sources, chunks_with_ids = context_bm25(query)
    
    sources = []
    if 'rag' in tools:
        sources.append(rag_sources)
    if 'bm25' in tools:
        sources.append(bm25_sources)

    fused = reciprocal_rank_fusion(sources)
    results = []
    sources = []
    print(fused)
    for i in range(5):
        for doc in chunks_with_ids:
            if doc.metadata.get("id", None) == fused[i][0]:
                results.append(doc.page_content)
                sources.append((fused[i][0], doc.page_content))

    context_text = "\n\n---\n\n".join([doc for doc in results])
    
    # if hist_prompt != "":
    #     model = ChatOpenAI(model_name= "gpt-4o", api_key=api_key)
    #     response = model.invoke(hist_prompt)
    #     hist_text = response.content
    # else:
    #     hist_text = "None"
    # PROMPT_TEMPLATE_HIST = f"CHAT HISTORY: \n {hist_text} {PROMPT_TEMPLATE}"
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, query=query)

    model = ChatOpenAI(model_name= "gpt-4o", api_key=api_key)
    response_text = model.invoke(prompt)

    # sources = [source, doc for source, doc in sources]
    formatted_response = response_text.content
    return formatted_response, sources
