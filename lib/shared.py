import os
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch

from langchain.chains import RetrievalQA
import tiktoken
from azure.search.documents.indexes.models import (
    SearchableField,
    SearchFieldDataType,
    SimpleField,
)


def get_az_chat_openai() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        openai_api_version="2023-05-15",
        azure_deployment=os.environ["MODEL_NAME"],
        streaming=True,
        temperature=0,
    )


def get_az_openai_embeddings() -> AzureOpenAIEmbeddings:
    embedding_model = os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
    return AzureOpenAIEmbeddings(deployment=embedding_model, chunk_size=256)


def get_vector_store(openai_embeddings: AzureOpenAIEmbeddings) -> AzureSearch:
    vector_store_address = os.environ["AZURE_SEARCH_ENDPOINT"]
    vector_store_pwd = os.environ["AZURE_SEARCH_KEY"]
    index_name = os.environ["AZURE_INDEX_NAME"]

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
        SearchableField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=len(openai_embeddings.embed_query("Text")),
        ),
        SearchableField(
            name="metadata",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
        # Additional field to store the title
        SearchableField(
            name="contract_name",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        # Additional field for filtering on document source
        SimpleField(
            name="doc_type",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        # Additional field for filtering on document source
        SimpleField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        # Additional data field for last doc update
        SimpleField(
            name="last_update",
            type=SearchFieldDataType.DateTimeOffset,
            searchable=True,
            filterable=True,
        ),
    ]

    return AzureSearch(
        azure_search_endpoint=vector_store_address,
        azure_search_key=vector_store_pwd,
        index_name=index_name,
        embedding_function=openai_embeddings.embed_query,
        fields=fields,
    )


def get_knowledge_folder() -> str:
    return os.environ["KNOWLEDGE_FOLDER"]


def get_staging_folder() -> str:
    return os.environ["STAGING_FOLDER"]


def is_debug_mode() -> bool:
    return os.environ["DEBUG"] == "1"


def num_token_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_qa_chain(az_chat_openai, vector_store, prompt_template, question, k, filters):
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": prompt}

    retriever = vector_store.as_retriever(
        search_kwargs={"k": k, "search_type": "hybrid", "filters": filters}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=az_chat_openai,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )

    result = qa_chain._call({"query": question})
    return result
