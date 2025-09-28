import neo4j
from neo4j_graphrag.llm import OpenAILLM as LLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings as Embeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation.graphrag import GraphRAG
from neo4j_graphrag.generation import RagTemplate
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
import os
import asyncio
import json

from dotenv import load_dotenv
# Load environment variables from .env file (optional)
load_dotenv()

#environment keys
api_key = os.getenv("OPENAI_API_KEY")

#pdf locations
KNOWLEDGE_BASE_PATH = "./knowledge_base/sample_pump_info.pdf"

database = os.getenv("NEO4J_DATABASE")
neo4j_driver = neo4j.GraphDatabase.driver(uri=os.environ["NEO4J_URI"],
                                          auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
                                          database=database)

ex_llm=OpenAILLM(
   model_name="gpt-4.1-mini",
   model_params={
       "response_format": {"type": "json_object"},
       "temperature": 0
   })

embedder = OpenAIEmbeddings()

vector_retriever = VectorRetriever(
   neo4j_driver,
   index_name="text_embeddings",
   embedder=embedder,
   return_properties=["text"],
)

def query_graphrag(query: str):
   print("Running GraphRAG...")
   llm = LLM(model_name="gpt-4o",  model_params={"temperature": 0.0})
   rag = GraphRAG(llm=llm, retriever=vector_retriever)

   rag_template = RagTemplate(template='''Answer the Question using the following Context. Only respond with information mentioned in the Context. Do not inject any speculative information not mentioned.

   # Question:
   {query_text}

   # Context:
   {context}

   # Answer:
   ''', expected_inputs=['query_text', 'context'])

   v_rag  = GraphRAG(llm=llm, retriever=vector_retriever, prompt_template=rag_template)

   # 4. Run
   rag_template = RagTemplate(template='''
   CONTEXT INFORMATION:
   {context}

   USER QUERY:
   "{query_text}"

   INSTRUCTIONS:
   - Answer the query using ONLY the information provided in the context above
   - If the context doesn't contain enough information to answer the query, clearly state this
   - Cite specific parts of the context when making claims
   - Be concise but complete in your response
   - If multiple interpretations are possible, acknowledge this

   ANSWER:
   ''', expected_inputs=['query_text', 'context'])

   v_rag  = GraphRAG(llm=llm, retriever=vector_retriever, prompt_template=rag_template)
   q = query
   # print("Query:\n", q)
   # print("n===========================n")
   # print(f"Vector Response: \n{v_rag.search(q, retriever_config={'top_k':5}).answer}")
   # print("n===========================n")
   response = v_rag.search(q, retriever_config={'top_k':5}).answer
   return response

# def graph_context():
   # query = "What is the failure mode of the equipment"
   # vector_res = vector_retriever.get_search_results(query_text = query,
   #       top_k=3)
   # for i in vector_res.records: print("====n" + json.dumps(i.data()['node']['text'], indent=4))

   # return vector_res