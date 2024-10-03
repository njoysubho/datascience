import os
import PyPDF2
import pinecone
from llama_index.core import Document, GPTVectorStoreIndex, ServiceContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores import PineconeVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.embeddings.mistralai import MistralAIEmbedding
from pinecone import Pinecone, ServerlessSpec
import time

def initialize_pinecone():
    pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'), environment='us-east-1')
    index_name = os.environ.get('PINECONE_INDEX_NAME')
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name, 
            dimension=1024,
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        ) 
    return pc.Index(index_name)

def initialize_vector_store(pinecone_index):
    return PineconeVectorStore(pinecone_index=pinecone_index)

def initialize_embedding_model():
    return MistralAIEmbedding(model_name="mistral-embed",
                                     api_key=os.environ.get('MISTRAL_API_KEY'),
                                     embed_batch_size=2)

def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def process_pdf_files(pdf_folder):
    pdf_paths = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    documents = []
    for pdf_path in pdf_paths:
        pdf_text = extract_text_from_pdf(pdf_path)
        documents.append(Document(text=pdf_text, metadata={'source': pdf_path}))

def parse_documents(documents):
    parser = SimpleNodeParser(text_splitter=TokenTextSplitter(chunk_size=512, chunk_overlap=50))
    return parser.get_nodes_from_documents(documents)

def initialize_service_context(embedding_model):
    return ServiceContext.from_defaults(embed_model=embedding_model)

def initialize_storage_context(vector_store):
    return StorageContext.from_defaults(vector_store=vector_store)

def upload_to_pinecone(nodes, service_context, storage_context, index_name):
    for i, node in enumerate(nodes):
        index = GPTVectorStoreIndex([node], storage_context=storage_context, service_context=service_context)
       
        # Pause for 1 second between each request
        if i < len(nodes) - 1:
            time.sleep(2)

        index.storage_context.persist()
print(f"Uploaded PDF vectors to Pinecone index '{index_name}'")

if __name__== "__main__":
    pinecone_index = initialize_pinecone()
    vector_store = initialize_vector_store(pinecone_index)
    embedding_model = initialize_embedding_model()
    documents = process_pdf_files('./docs')
    nodes = parse_documents(documents)
    service_context = initialize_service_context(embedding_model)
    storage_context = initialize_storage_context(vector_store)
    upload_to_pinecone(nodes, service_context, storage_context, pinecone_index.name)
    

