from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langchain.prompts import PromptTemplate
from operator import itemgetter

from lib.shared import get_az_chat_openai, get_az_openai_embeddings, get_vector_store

import chainlit as cl


def format_docs(docs):
    print(docs)
    return "\n\n".join(doc.page_content for doc in docs)


@cl.on_chat_start
async def on_chat_start():
    model = get_az_chat_openai()
    vector_store = get_vector_store(get_az_openai_embeddings())
    retriever = vector_store.as_retriever()
    
    custom_assistant_template = """
You are an assistant chatbot named "Copilot". Your expertise is 
exclusively in providing information about provided context.
You do not provide information outside of this scope. 
Context: {context}
Question: {question}
Answer:"""

    custom_assistant_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=custom_assistant_template
    )

    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs, 
            "question": itemgetter("question")
        }
        | custom_assistant_prompt
        | model
        | StrOutputParser()
    )
    cl.user_session.set("runnable", rag_chain)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    for chunk in await cl.make_async(runnable.stream)(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
