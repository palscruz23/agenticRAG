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
# NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
# NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
# NEO4J_URI = os.environ["NEO4J_URI"]

#pdf locations
KNOWLEDGE_BASE_PATH = "knowledge base/sample_pump_info.pdf"

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

prompt_template = '''
You are a reliability engineer working to develop knowledge graph for specific equipment maintenance by structuring the information from the given input text into a knowledge graph that can be used to determine the failure mode, cause, and effect of the equipment.

Extract the entities (nodes) and specify their type from the following Input text.
Also extract the relationships between these nodes. the relationship direction goes from the start node to the end node. 

Return result as JSON using the following format:
{{"nodes": [ {{"id": "0", "label": "the type of entity", "properties": {{"name": "name of entity" }} }}],
  "relationships": [{{"type": "TYPE_OF_RELATIONSHIP", "start_node_id": "0", "end_node_id": "1", "properties": {{"details": "Description of the relationship"}} }}] }}

- Use only the information from the Input text. Do not add any additional information.  
- If the input text is empty, return empty Json. 
- Make sure to create as many nodes and relationships as needed to offer rich medical context for further research.
- An AI knowledge assistant must be able to read this graph and immediately understand the context to inform detailed research questions. 
- Multiple documents will be ingested from different sources and we are using this property graph to connect information, so make sure entity types are fairly general. 

Assign a unique ID (string) to each node, and reuse it to define relationships.
Do respect the source and target node types for relationship and
the relationship direction.

Do not return any additional information other than the JSON in it.

Input text:
{text}
'''

node_labels = ["Equipment", "Failure Mode", "Failure Cause", "Failure Effect", "Document", "Chunk"]

# define relationship types
rel_types = ["FAILS_DUE_TO", "CAUSED_BY", "RESULTS"]

# POTENTIAL_SCHEMA = [
#     ("Equipment", "FAILS_DUE_TO", "Failure Mode"),
#     ("Failure Mode", "CAUSED_BY", "Failure Cause"),
#     ("Failure Mode", "RESULTS", "Failure Effect")
# ]

def main():
   print("Building KG...")
   kg_builder_pdf = SimpleKGPipeline(
      llm=ex_llm,
      driver=neo4j_driver,
      text_splitter=FixedSizeSplitter(chunk_size=5000, chunk_overlap=500),
      embedder=embedder,
      entities=node_labels,
      relations=rel_types,
      prompt_template=prompt_template,
      from_pdf=True
   )
   asyncio.run(kg_builder_pdf.run_async(file_path=KNOWLEDGE_BASE_PATH))
   print("KG built and stored in Neo4j database.")

   # 2. KG Retriever
   print("Retrieving KG...")
   from neo4j_graphrag.indexes import create_vector_index

   create_vector_index(neo4j_driver, name="text_embeddings", label="Chunk",
                     embedding_property="embedding", dimensions=1536, similarity_fn="cosine")

if __name__ == "__main__":
   main()