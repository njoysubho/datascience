import os
from llama_index.llms.mistralai import MistralAI
from llama_index.core import VectorStoreIndex, ServiceContext,StorageContext,Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
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


def initialize_service_context(embedding_model):
    return ServiceContext.from_defaults(embed_model=embedding_model)

def initialize_storage_context(vector_store):
    return StorageContext.from_defaults(vector_store=vector_store)




if __name__=="__main__":
    pinecone_index = initialize_pinecone()
    vector_store = initialize_vector_store(pinecone_index)
    embedding_model = initialize_embedding_model()
    Settings.embed_model = embedding_model
    Settings.llm = MistralAI(model="mistral-small",api_key=os.environ.get('MISTRAL_API_KEY'))
    index = VectorStoreIndex.from_vector_store(vector_store)
    time.sleep(10)
    response = index.as_query_engine().query("")
    print(response)
    

