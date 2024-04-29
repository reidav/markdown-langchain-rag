import sys
from dotenv import load_dotenv
from shared import (
    get_az_chat_openai,
    get_az_openai_embeddings,
    get_vector_store,
    get_qa_chain,
)

if __name__ == "__main__":
    load_dotenv()

    az_chat_openai = get_az_chat_openai()
    az_openai_embeddings = get_az_openai_embeddings()
    vector_store = get_vector_store(az_openai_embeddings)

    try:
        prompt_template = """
            Given the following extracted parts of a long document and a question, answer the question without explaining why you picked it.
            When you are being asked to mention parts of the text, do not translate them, just mention them as they are.
            {context}

            My QUESTION is : {question}
            """

        question = sys.argv[1] if len(sys.argv) > 1 else None
        if not question:
            raise Exception("No question provided.")

        # Async Mode / Multiple questions / Parallel processing should be in async def main():
        # tasks = [
        #     get_qa_chain(az_chat_openai, vector_store, prompt_template, question, k=5),
        #     ...,  # Add more questions here
        # ]
        # results = await asyncio.gather(*tasks)

        # Single question
        response = get_qa_chain(
            az_chat_openai, vector_store, prompt_template, question, k=5, filters={}
        )

        print(response)

    except Exception as e:
        raise e
