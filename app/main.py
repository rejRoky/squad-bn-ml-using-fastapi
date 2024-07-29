from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from redis import Redis
import psycopg2
from fastapi.responses import RedirectResponse

from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import pandas as pd

# Load dataset
from datasets import load_dataset

dataset = load_dataset("csebuetnlp/squad_bn")

train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])
validation_df = pd.DataFrame(dataset['validation'])

merged_df = pd.concat([train_df, test_df, validation_df], ignore_index=True)

# Load the precomputed embeddings if they exist, otherwise compute them
try:
    corpus_embeddings = torch.tensor(np.load("squad_bn_emb_e5_large_instruct.npy"))
except FileNotFoundError:
    model = SentenceTransformer("intfloat/multilingual-e5-large")
    passages_bn = merged_df['context']
    corpus_embeddings = model.encode(passages_bn, batch_size=64, convert_to_tensor=True, show_progress_bar=True)
    np.save("squad_bn_emb_e5_large_instruct.npy", corpus_embeddings.cpu().numpy())

app = FastAPI()

# Load the SentenceTransformer model
model = SentenceTransformer("intfloat/multilingual-e5-large")


# Define the request and response models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    hits: list


@app.get("/")
async def read_root():
    # return {"message": "Visit /docs for documentation of the available endpoints."}
    return RedirectResponse(url='/docs')


# Define the endpoint
@app.post("/query", response_model=QueryResponse)
def search(query_request: QueryRequest):
    query = query_request.query
    top_k = query_request.top_k

    if not torch.cuda.is_available():
        raise HTTPException(status_code=500, detail="No GPU found. Please add GPU to your notebook")

    # Encode the query
    question_embedding = model.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding.cuda()

    # Perform semantic search
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)[0]

    # Format the response
    response_hits = [
        {
            "score": hit['score'],
            "context": merged_df['context'][hit['corpus_id']],
        }
        # 1st hit show only
        for hit in hits[0:1]
    ]

    return QueryResponse(hits=response_hits)


@app.get("/mongodb")
async def read_mongodb():
    try:
        client = MongoClient("mongodb://mongodb:27017/")
        db = client["test-database"]
        collection = db["test-collection"]
        collection.insert_one({"status": "working"})
        document = collection.find_one({}, {"_id": 0})
        return {"mongodb": document}
    except Exception as error:
        return {"mongodb": str(error)}


@app.get("/redis")
async def read_redis():
    try:
        redis = Redis(host="redis", port=6379)
        redis.set("status", "working")
        value = redis.get("status")
        return {"redis": value.decode("utf-8")}
    except Exception as error:
        return {"redis": str(error)}


@app.get("/postgres")
async def read_postgres():
    conn = None
    try:
        conn = psycopg2.connect(
            host="postgres",
            port=5432,
            database="vpa_psql_db",
            user="eblict_vpa",
            password="eblict_vpa_admin",
        )
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS test_table;")
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS test_table (id SERIAL PRIMARY KEY, name VARCHAR);"
        )
        cursor.execute("INSERT INTO test_table (name) VALUES ('success');")
        cursor.execute("SELECT * FROM test_table")
        record = cursor.fetchone()
        return {"postgres": record}
    except (Exception, psycopg2.Error) as error:
        return {"postgres": str(error)}
    finally:
        if conn:
            cursor.close()
            conn.close()
            print("PostgreSQL connection is closed")
