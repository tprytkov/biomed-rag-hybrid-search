\# Biomedical Hybrid-RAG API Integration Examples



This reference manual provides production code snippets for querying the live system gateway from external applications.



\## 1. Interactive API Reference Dashboard

When the server application is active locally, you can view, test, and interact with the automatically generated OpenAPI web interface by visiting:

\* \*\*Swagger UI Documentation Link\*\*: \[http://localhost:8000/docs](http://localhost:8000/docs)



\---



\## 2. Command Line Interface Execution (cURL)

To run a quick terminal test loop without using Python scripts, use this standard request pattern:



```bash

curl -X 'POST' \\

&#x20; 'http://localhost:8000/v1/query' \\

&#x20; -H 'accept: application/json' \\

&#x20; -H 'Content-Type: application/json' \\

&#x20; -d '{

&#x20; "query": "What mutations are targetable by third-generation kinase inhibitors?",

&#x20; "top\_k": 2

}'

```



\---



\## 3. Python Integration Script (`requests`)

This production-ready implementation handles structured data querying, checking for error statuses, and displaying response results:



```python

import requests

import json



def query\_biomed\_rag(prompt: str, results\_count: int = 2):

&#x20;   url = "http://localhost:8000/v1/query"

&#x20;   headers = {

&#x20;       "accept": "application/json",

&#x20;       "Content-Type": "application/json"

&#x20;   }

&#x20;   payload = {

&#x20;       "query": prompt,

&#x20;       "top\_k": results\_count

&#x20;   }



&#x20;   try:

&#x20;       response = requests.post(url, headers=headers, json=payload)

&#x20;       

&#x20;       if response.status\_code == 200:

&#x20;           data = response.json()

&#x20;           print("=== GENERATED RESPONSE ===")

&#x20;           print(f"{data\['answer']}\\n")

&#x20;           

&#x20;           print("=== VERIFIABLE EVIDENCE BLOCKS RETRIEVED ===")

&#x20;           for rank, context in enumerate(data\['retrieved\_context'], 1):

&#x20;               print(f"Rank {rank} | Chunk: {context\['chunk\_id']} | Score: {context\['score']:.4f}")

&#x20;               print(f"Text: {context\['text']}\\n")

&#x20;       else:

&#x20;           print(f"API Error ({response.status\_code}): {response.text}")

&#x20;           

&#x20;   except requests.exceptions.ConnectionError:

&#x20;       print("Network Error: Ensure your FastAPI application server is running on port 8000.")



if \_\_name\_\_ == "\_\_main\_\_":

&#x20;   query\_biomed\_rag("kinase inhibitor resistance mutations")

```



\---



\## 4. JavaScript / TypeScript Integration Blueprint (`fetch`)

Use this implementation structure to connect a custom web frontend UI dashboard to your server logic:



```javascript

async function sendBiomedicalQuery(userQuestion) {

&#x20;   const endpoint = 'http://localhost:8000/v1/query';

&#x20;   

&#x20;   const requestOptions = {

&#x20;       method: 'POST',

&#x20;       headers: {

&#x20;           'accept': 'application/json',

&#x20;           'Content-Type': 'application/json'

&#x20;       },

&#x20;       body: JSON.stringify({

&#x20;           query: userQuestion,

&#x20;           top\_k: 2

&#x20;       })

&#x20;   };



&#x20;   try {

&#x20;       const response = await fetch(endpoint, requestOptions);

&#x20;       if (!response.ok) {

&#x20;           throw new Error(`HTTP network exception error: Status ${response.status}`);

&#x20;       }

&#x20;       

&#x20;       const resultPayload = await response.json();

&#x20;       console.log("Answer Output:", resultPayload.answer);

&#x20;       console.log("Source Citations:", resultPayload.retrieved\_context);

&#x20;       return resultPayload;

&#x20;   } catch (error) {

&#x20;       console.error("Pipeline failure connection error execution:", error);

&#x20;   }

}

```



