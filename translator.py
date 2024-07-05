from config import api_key, endpoint
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

model = AzureChatOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    openai_api_version="2023-05-15",
    azure_deployment="name-of-your-deployment",
    temperature=0.5,
    streaming=True,
)
parser = StrOutputParser()

system_template = "Translate the following into {language}, then write about the history of this language in english."
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

chain = prompt_template | model | parser
user_language = input("What language to translate into? ")
user_input = input("Enter text to translate: ")
print(chain.invoke({"language": user_language, "text": user_input}))
