from fastapi import FastAPI, HTTPException,UploadFile, File
from pydantic import BaseModel
import requests
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from neo4j import GraphDatabase
from transformers import BertTokenizer, BertModel
import torch
import fitz  # PyMuPDF
import uuid
import tempfile
import os
import io
import pandas as pd
import logging
from typing import List
from dotenv import load_dotenv

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


# MongoDB setup
mongo_client = MongoClient(MONGO_DB_URL)
db = mongo_client["pdf_db"]
chunks_collection = db["chunks"]

# Neo4j setup
neo4j_driver = GraphDatabase.driver(NEO_DB_HOST, auth=(NEO_DB_USER, NEO_DB_PASSWORD))


# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

class ChunkEmbedding(BaseModel):
    chunk_id: str
    text: str
    embedding: List[float]
    doc_id: str

# Utility function to create embeddings
def get_embeddings(text: str):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).flatten().tolist()
    return embeddings

# Utility function to save relationship in Neo4j
# def save_relationship(doc_id: str, chunk_ids: List[str]):
#     with neo4j_driver.session() as session:
#         session.run("CREATE (d:Document {id: $doc_id})", doc_id=doc_id)
#         for chunk_id in chunk_ids:
#             session.run("CREATE (c:Chunk {id: $chunk_id})", chunk_id=chunk_id)
#             session.run("MATCH (d:Document {id: $doc_id}), (c:Chunk {id: $chunk_id}) "
#                         "CREATE (d)-[:CONTAINS]->(c)", doc_id=doc_id, chunk_id=chunk_id)

def save_to_neo4j(doc_id: str, chunk_ids: List[str],chunk_text:str,chunk_embedding:List[float]):
     with neo4j_driver.session() as session:
         print(f"Creating doc_id: {doc_id} ")
         session.run("CREATE (d:Document {doc_id: $doc_id})", doc_id=doc_id)
        #  for chunk_id in chunk_ids:
        #      session.run("CREATE (ci:ChunkId {chunk_id: $chunk_id})", chunk_id=chunk_id)
        #      session.run("MATCH (d:Document {doc_id: $doc_id}), (c:Chunk {chunk_id: $chunk_id}) "
        #                  "CREATE (d)-[:CONTAINS]->(ci)", doc_id=doc_id, chunk_id=chunk_id)
             
         for chunk_id,chunk_text,chunk_embedding in zip(chunk_ids, chunk_text,chunk_embedding):
             print(f"Creating ChunkText with id: {chunk_id} and text: {chunk_text} and embedding: {chunk_embedding}")
             
             session.run("CREATE (ci:ChunkId {chunk_id: $chunk_id})", chunk_id=chunk_id)
             session.run("MATCH (d:Document {doc_id: $doc_id}), (ci:ChunkId {chunk_id: $chunk_id}) "
                         "MERGE (d)-[:CONTAINS]->(ci)", doc_id=doc_id, chunk_id=chunk_id)
             
             session.run("CREATE (ct:ChunkText {chunk_text: $chunk_text})", chunk_text=chunk_text)
             session.run("MATCH (ci:ChunkId {chunk_id: $chunk_id}), (ct:ChunkText {chunk_text: $chunk_text})"
                         "MERGE (ci)-[:HAS_TEXT]->(ct)", chunk_id=chunk_id,chunk_text=chunk_text)
             
             
             session.run("CREATE (ce:ChunkEmbedding {chunk_embedding: $chunk_embedding})", chunk_embedding=chunk_embedding)
             session.run("MATCH (ci:ChunkId {chunk_id: $chunk_id}), (ce:ChunkEmbedding {chunk_embedding: $chunk_embedding})"
                         "MERGE (ci)-[:HAS_EMBEDDING]->(ce)", chunk_id=chunk_id,chunk_embedding=chunk_embedding)
             
             session.run("MATCH (ct:ChunkText {chunk_text: $chunk_text}), (ce:ChunkEmbedding {chunk_embedding: $chunk_embedding})"
                          "MERGE (ce)-[:THIS_EMBEDDING_IS_FOR]->(ct)", chunk_text=chunk_text,chunk_embedding=chunk_embedding)
            #  session.run("CREATE (ci)-[:HAS_TEXT]->(ct)", chunk_id=chunk_id,chunk_text=chunk_text)
            #  session.run("CREATE (ci)-[:CONTAINS]->(e)", chunk_id=chunk_id,embedding=embedding)
            #  session.run("CREATE (ci)-[:CONTAINS]->(ct)", embedding=embedding, chunk_id=chunk_id)

# Utility function to extract text from a PDF file
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    pdf_document = fitz.open(file_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# Utility function to break text into chunks
def break_text_into_chunks(text: str, chunk_size: int = 512) -> List[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

@app.post("/uploadPDF/")
async def upload(file: UploadFile = File(...), chunk_size: int = 512):
    try:
        doc_id = str(uuid.uuid4())
        chunk_ids = []

        # Create a temporary file to handle the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Extract text from PDF
        text = extract_text_from_pdf(file_path=temp_file_path)
        logger.info("Extracted text from PDF")

        # Break text into chunks
        chunks = break_text_into_chunks(text, chunk_size)
        logger.info(f"Text broken into {len(chunks)} chunks")

        # Insert chunks and embeddings into MongoDB
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)
            embedding = get_embeddings(chunk)
            chunk_embedding = ChunkEmbedding(chunk_id=chunk_id, text=chunk, embedding=embedding, doc_id=doc_id)
            #chunks_collection.insert_one(chunk_embedding.dict())
            logger.info(f"Inserted chunk {chunk_id} into MongoDB")

        # Save relationships in Neo4j
        #save_relationship(doc_id, chunk_ids)
        save_to_neo4j(doc_id, chunk_ids,chunks,embedding)
        logger.info(f"Saved relationships in Neo4j for doc_id {doc_id}" and {chunk_ids})

        # Clean up temporary file
        os.remove(temp_file_path)

        return JSONResponse(content={"doc_id": doc_id})

    except Exception as e:
        logger.error(f"Error processing file upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))



def save_chunks(df, chunk_size, output_dir):
    """Splits the DataFrame into chunks and saves them as separate CSV files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)
    
    for i in range(num_chunks):
        chunk = df[i*chunk_size : (i+1)*chunk_size]
        chunk.to_csv(f"{output_dir}/chunk_{i+1}.csv", index=False)

@app.post("/uploadCSV/")
async def upload_file(file: UploadFile = File(...), chunk_size: int = 1000):
    """API endpoint to upload, read, chunk, and save CSV file."""
    contents = await file.read()
    #df = pd.read_csv(pd.compat.StringIO(contents.decode('utf-8')))
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

    
    output_dir = "output_chunks"
    save_chunks(df, chunk_size, output_dir)
    
    return {"message": f"CSV file uploaded and split into chunks of {chunk_size} rows each.", "output_dir": output_dir}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
