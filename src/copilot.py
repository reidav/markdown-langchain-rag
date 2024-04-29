from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from shared import get_az_chat_openai, get_az_openai_embeddings, get_vector_store

import chainlit as cl


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@cl.on_chat_start
async def on_chat_start():
    model = get_az_chat_openai()
    vector_store = get_vector_store(get_az_openai_embeddings())
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given the following extracted parts of a long document and a question, answer the question without explaining why you picked it. \
                When you are being asked to mention parts of the text, do not translate them, just mention them as they are.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    cl.user_session.set("runnable", rag_chain)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
