import joblib 
import json 
import os 
import requests
import pandas as pd 
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity


def create_embeddings(text_list):
    r=requests.post("http://localhost:11434/api/embed",
                json={
                    "model":"bge-m3",
                    "input": text_list
                } 
             
                )
    embedding = r.json()['embeddings']
    return embedding

# a = create_embeddings(["Hello brother","My name is shubham"])
# print(a)
newjsons = os.listdir("newjsons")
chunk_id=0
my_dict = []
for json_file in newjsons:
    with open(f"newjsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    embedding = create_embeddings([c['text'] for c in content['chunks']])
    for i,chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embedding[i]
        chunk_id+=1
        my_dict.append(chunk)
    # print(chunk)


df = pd.DataFrame.from_records(my_dict)
# print(df)

# save the embeddings 

joblib.dump(df,'embeddings.joblib')
