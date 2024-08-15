from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureChatOpenAI
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_VERSION = os.environ.get('AZURE_OPENAI_VERSION')


embeddings = download_hugging_face_embeddings()

index_name="medical-chatbot"

vectorstore=PineconeVectorStore.from_existing_index(index_name, embeddings)


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

llm = AzureChatOpenAI(
    openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    model_name="newest",  # Ensure this matches your deployment
    azure_endpoint="https://chantestaoi.openai.azure.com/openai/deployments/newest/chat/completions?api-version=2024-02-15-preview",
    api_version="2024-02-15-preview",
  # Ensure this matches your deployment
    temperature=0.7,
    max_tokens=512
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)