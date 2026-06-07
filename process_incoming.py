import pandas as pd 
import numpy as np 
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import requests

def create_embeddings(text_list):
    r=requests.post("http://localhost:11434/api/embed",
                json={
                    "model":"bge-m3",
                    "input": text_list
                } 
             
                )
    embedding = r.json()['embeddings']
    return embedding
def inferencef(prompt):
        r=requests.post("http://localhost:11434/api/generate",
                json={
                    # "model":"deepseek-r1",
                    "model":"llama3.2",
                    "prompt": prompt,
                    "stream": False
                } 
                )
        response = r.json()
        # print(response)
        return response
    

df = joblib.load("embeddings.joblib")
incoming_query = input("Ask a Question:")
question_embedding = create_embeddings([incoming_query])[0]

# print(question_embedding)

# find similiarity of question embeddings with other embeddings
# i have to studey for it 
# it should be imp for this course 

similiarities = cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()
# print(similiarities) # it shows the similiarity in embeddings form
top_result = 3
max_indx = similiarities.argsort()[::-1][0:top_result]
# print(max_indx) # it gives us the top matched index which result is relevent 
new_df = df.loc[max_indx]
# print(new_df[['title','number','text']]) # this is the data which is store in new dataframe

# now we are going to iterate this new_df for gettings result 
prompt = f"""
You are an assistant for the Sigma Web Development Course, which teaches web development in a clear and straightforward manner.  
Below are structured subtitle chunks from course videos. Each chunk includes:  
- video title  
- video number  
- start and end time (in seconds)  
- the spoken text during that segment  

Course Data:
{new_df[['title', 'number', 'start', 'end', 'text']].to_json(orient='records')}

------------------------------------------------------------

User Query: "{incoming_query}"

Your task:
1. Answer naturally in a human way — don't mention JSON, subtitles, or data sources.  
2. Identify which video(s) and time range(s) cover the topic related to the user’s question.  
3. Tell the user what part of the topic is explained, and guide them to the correct video and timestamp (e.g., “You can learn this in Video 4 around 3:20 minutes”).  
4. If the question is not related to the Sigma Web Development course, politely respond that you can only answer questions related to the course.

Make your answer clear, concise, and helpful for a beginner learning web development.
"""

with open("prompt.txt","w") as f:
    f.write(prompt)
    # print("your prompt is ready!!")
    print("Thinking...")

response = inferencef(prompt)['response']
print(response)
with open("response.txt","w") as f:
    f.write(response)

# for index, item in new_df.iterrows():
#     print(index, item['title'], item['number'], item['text'])
