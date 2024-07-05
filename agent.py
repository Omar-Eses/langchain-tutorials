import json
from config import api_key, endpoint, tavily_api_key
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

model = AzureChatOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    openai_api_version="2023-05-15",
    azure_deployment="name-of-your-deployment",
    temperature=0.5,
    streaming=True,
)

# Ensure you pass the parameter correctly as expected by your Pydantic model
search = TavilySearchAPIWrapper(tavily_api_key=tavily_api_key)
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=2)
result = tavily_tool.invoke("what is the weather in SF?")
print(json.dumps(result[0]["content"], indent=2))

loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)


vector = FAISS.from_documents(
    documents,
    AzureOpenAIEmbeddings(
        api_key=api_key, azure_endpoint=endpoint, azure_deployment="name-of-your-deployment"
    ),
)
retriever = vector.as_retriever()

print(len(documents))
print(retriever.invoke("how to upload a dataset")[0])
