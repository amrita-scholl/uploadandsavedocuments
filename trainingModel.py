from fastapi import FastAPI, HTTPException
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import logging
import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split


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
from transformers import BertTokenizer

# Neo4j setup
neo4j_driver = GraphDatabase.driver(NEO_DB_HOST, auth=(NEO_DB_USER, NEO_DB_PASSWORD))

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def retrieve_documents_from_neo4j():
    query = """
    MATCH (d:Document)-[:CONTAINS]->(ci:ChunkId)-[:HAS_TEXT]->(ct:ChunkText)
    RETURN d.doc_id AS document_id, 
           collect(ct.chunk_text) AS chunk_texts
    """
    with neo4j_driver.session() as session:
        result = session.run(query)
        documents = []
        for record in result:
            document = {
                "document_id": record["document_id"],
                "chunk_texts": record["chunk_texts"]
            }
            documents.append(document)
        return documents

# Example usage
documents = retrieve_documents_from_neo4j()

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_texts(documents):
    texts = []
    for document in documents:
        texts.extend(document['chunk_texts'])
    return texts

# Example usage
texts = preprocess_texts(documents)

# Prepare the dataset
labels = [0] * len(texts)  # Example: Dummy labels, replace with actual labels if available

# Tokenize the dataset
encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

# Create a dataset class
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Split the dataset into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(encodings['input_ids'], labels, test_size=0.2)

train_encodings = {key: val[:len(train_texts)] for key, val in encodings.items()}
val_encodings = {key: val[len(train_texts):] for key, val in encodings.items()}

train_dataset = TextDataset(train_encodings, train_labels)
val_dataset = TextDataset(val_encodings, val_labels)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Check if GPU is available and set the device accordingly
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="steps",     # Evaluate every `logging_steps`
    save_steps=10,                   # Save checkpoint every `save_steps`
    save_total_limit=2,              # Only keep the last 2 checkpoints
    load_best_model_at_end=True,     # Load the best model at the end of training
)

# Create Trainer instance
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model_save_path = "./trained_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model and tokenizer saved to {model_save_path}")