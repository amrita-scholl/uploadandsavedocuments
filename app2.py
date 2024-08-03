from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import GraphDatabase
import torch
import logging
from dotenv import load_dotenv
import os
from typing import List
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification


load_dotenv()

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get environment variables
MONGO_DB_URL=os.getenv("MONGO_DB_URL")
NEO_DB_HOST = os.getenv("NEO_DB_HOST")
NEO_DB_USER = os.getenv("NEO_DB_USER")
NEO_DB_PASSWORD = os.getenv("NEO_DB_PASSWORD")

# Neo4j setup
neo4j_driver = GraphDatabase.driver(NEO_DB_HOST, auth=(NEO_DB_USER, NEO_DB_PASSWORD))

# Load the trained model and tokenizer
model_path = "./trained_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

class QueryRequest(BaseModel):
    query: str

def retrieve_chunks_from_neo4j():
    query = """
    MATCH (d:Document)-[:CONTAINS]->(ci:ChunkId)-[:HAS_TEXT]->(ct:ChunkText)
    OPTIONAL MATCH (ci)-[:HAS_EMBEDDING]->(ce:ChunkEmbedding)
    RETURN d.doc_id AS document_id, 
           ci.chunk_id AS chunk_id, 
           ct.chunk_text AS chunk_text,
           ce.chunk_embedding AS chunk_embedding
    """
    with neo4j_driver.session() as session:
        result = session.run(query)
        chunks = []
        for record in result:
            chunk = {
                "document_id": record["document_id"],
                "chunk_id": record["chunk_id"],
                "chunk_text": record["chunk_text"],
                "chunk_embedding": record.get("chunk_embedding", None)  # Use .get() to handle optional value
            }
            chunks.append(chunk)
        return chunks

    
@app.post("/answer/")
async def answer_query(request: QueryRequest):
    query_text = request.query
    print("query_text : ",query_text)
    # Retrieve chunks from Neo4j
    chunks = retrieve_chunks_from_neo4j()
    print("chunks : ",chunks)
    # Preprocess the query and chunks
    inputs = tokenizer([query_text] + [chunk['chunk_text'] for chunk in chunks], padding=True, truncation=True, return_tensors="pt")
    
    # Perform inference
    outputs = model(**inputs)
    logits = outputs.logits
    
    # Calculate the scores for the chunks
    scores = logits[0][1:].tolist()  # Ignore the score of the query itself
    
    # Find the chunk with the highest score
    max_score_index = scores.index(max(scores))
    best_chunk = chunks[max_score_index]
    
    response = {
        "document_id": best_chunk["document_id"],
        "chunk_id": best_chunk["chunk_id"],
        "chunk_text": best_chunk["chunk_text"]
    }
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)