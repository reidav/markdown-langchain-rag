import os
import glob
import base64
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, ContentFormat
from lib.shared import get_knowledge_folder, get_staging_folder


def get_doc_as_markdown(pdf_filepath, output_filepath):
    endpoint = os.environ["DOCUMENT_INTELLIGENCE_ENDPOINT"]
    key = os.environ["DOCUMENT_INTELLIGENCE_API_KEY"]

    document_intelligence_client = DocumentIntelligenceClient(
        endpoint, AzureKeyCredential(key)
    )

    with open(pdf_filepath, "rb") as f:
        bytes_source = base64.b64encode(f.read()).decode("utf-8")

        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout",
            AnalyzeDocumentRequest(bytes_source=bytes_source),
            output_content_format=ContentFormat.MARKDOWN,
        )
        result = poller.result()
        markdown_output = result.content

        with open(output_filepath, "w") as output_file:
            output_file.write(markdown_output)


if __name__ == "__main__":
    load_dotenv()

    source_folder = os.path.join(get_knowledge_folder(), "*.pdf")
    print(f"Searching for PDF files in: {source_folder}")
    pdf_files = glob.glob(source_folder)
    print(f"Found {len(pdf_files)} PDF files")
    for pdf_file in pdf_files:
        output_filepath = os.path.join(
            get_staging_folder(), f"{os.path.basename(pdf_file)}.md"
        )
        get_doc_as_markdown(pdf_file, output_filepath)
        print(f"Markdown file saved at: {output_filepath}")
