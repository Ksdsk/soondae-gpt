import openai
import os

from flask import Flask
from flask import request

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

# SETUP
app = Flask(__name__)

loader = DirectoryLoader("data/")
index = VectorstoreIndexCreator().from_loaders([loader])
chain = ConversationalRetrievalChain.from_llm(
    llm = ChatOpenAI(model="gpt-3.5-turbo"),
    retriever = index.vectorstore.as_retriever(search_kwargs={"k": 1})
)

chat_history = []
# API ENDPOINTS
@app.route("/")
def hello():
    return "hello from Flask!"

@app.route("/gpt", methods=["POST"])
def gpt():
    # Ask GPT
    prompt = request.json["prompt"]
    completion = chain({
        "question": prompt, 
        "chat_history": chat_history
    })

    # After
    chat_history.append((prompt, completion["answer"]))
    result = {
        "completion": completion["answer"],
        "received": True
    }

    return result, 200, {'ContentType':'application/json'}