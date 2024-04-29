from enum import Enum
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter

from shared import get_az_openai_embeddings, get_staging_folder, get_vector_store

if __name__ == "__main__":
    print("Loading Azure resources...")
    load_dotenv()
    embeddings = get_az_openai_embeddings()
    vector_store = get_vector_store(embeddings)

    print("Loading staging...")
    loader = DirectoryLoader(
        get_staging_folder(),
        glob=f"*.md",
        loader_cls=TextLoader,
        recursive=True,
        loader_kwargs={"autodetect_encoding": True},
    )

    documents = loader.load()

    print("Splitting documents ...")
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]

    for doc in documents:
        path = doc.metadata["source"]
        filename = os.path.basename(os.path.dirname(path))
        source = os.path.basename(path)

        print(f"Splitting document {path} ...")
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, return_each_line=False
        )

        md_header_splits = markdown_splitter.split_text(doc.page_content)

        for chunk in md_header_splits:
            chunk.metadata["source"] = source

        vector_store.add_documents(documents=md_header_splits)
