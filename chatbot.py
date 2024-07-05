from config import api_key, endpoint
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

model = AzureChatOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    openai_api_version="2024-02-01",
    azure_deployment="name-of-your-deployment",
    temperature=0.2,
    streaming=True,
)
parser = StrOutputParser()
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions related to user questions in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config = {"configurable": {"session_id": "abc11"}}

response = with_message_history.invoke(
    {"messages": [HumanMessage(content="my name is omar?")], "language": "English"},
    config=config,
)
print(response.content)
response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="whats my name?")],
        "language": "English",
    },
    config=config,
)
print(response.content)


def filter_messages(messages, k=10):
    return messages[-k:]


chain = (
    RunnablePassthrough.assign(
        messages=lambda x: filter_messages(messages=x["messages"], k=10)
    )
    | prompt
    | model
)

messages = [
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like Chocolate ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config = {"configurable": {"session_id": "abc11"}}

response = with_message_history.invoke(
    {
        "messages": messages
        + [HumanMessage(content="What is the last question i asked you?")],
        "language": "English",
    },
    config=config,
)

response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="what is my name?")],
        "language": "English",
    },
    config=config,
)

print(response.content)
config = {"configurable": {"session_id": "abc15"}}
for r in with_message_history.stream(
    {
        "messages": [
            HumanMessage(content="hi! I'm omar. tell me a very lame dad joke")
        ],
        "language": "English",
    },
    config=config,
):
    print(r.content, end="")
