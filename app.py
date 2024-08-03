from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi import FastAPI
import logging
from typing import List
from dotenv import load_dotenv
import os
import openai

load_dotenv()

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get environment variables
NEO_DB_HOST = os.getenv("NEO_DB_HOST")
NEO_DB_USER = os.getenv("NEO_DB_USER")
NEO_DB_PASSWORD = os.getenv("NEO_DB_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Neo4j Connection Class
class Neo4jConnection:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self._driver.close()

    def query(self, query, parameters=None):
        with self._driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]
        
# Neo4j setup
neo4j_conn = Neo4jConnection(NEO_DB_HOST, NEO_DB_USER, NEO_DB_PASSWORD)

# Tokenizer and model setup for BART
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")

def get_chunks(query):
    neo4j_query = """
    MATCH (d:Document)-[:CONTAINS]->(ci:ChunkId)-[:HAS_TEXT]->(ct:ChunkText)
    OPTIONAL MATCH (ci)-[:HAS_EMBEDDING]->(ce:ChunkEmbedding)
    RETURN d.doc_id AS document_id, 
           ci.chunk_id AS chunk_id, 
           ct.chunk_text AS chunk_text,
           ce.chunk_embedding AS chunk_embedding
    """
    results = neo4j_conn.query(neo4j_query, {"query": query})
    chunks = [record["chunk_text"] for record in results]
    return chunks

def generate_response(query):
    chunks = get_chunks(query)
    context = " ".join(chunks)
    combined_input = context + "\n\n" + query
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": combined_input}
        ],
        max_tokens=150
    )
    return response['choices'][0]['message']['content'].strip()
@app.get("/chat")
async def chat(query: str):
    response = generate_response(query)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
