from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct

def connect_qdrant(host="localhost", port=6333, timeout=100000.0):
    client = QdrantClient(host=host, port=port, timeout=timeout)
    return client

def create_collection(client, collection_name, vector_size):
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance="Cosine")
    )

def upload_vectors(client, collection_name, embeddings, texts, labels, batch_size=100):
    """Veri noktalarını Qdrant koleksiyonuna batch olarak yükler."""
    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                "text": texts[i],
                "level_0": labels[i][0],
                "level_1": labels[i][1],
                "level_2": labels[i][2],
                "level_3": labels[i][3],
                "level_4": labels[i][4],
                "label": labels[i][5],
            }
        )
        for i in range(len(embeddings))
    ]
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)

def search_collection(client, collection_name, query_vector, limit=3):
    search_result = client.search(
        collection_name=collection_name, 
        query_vector=query_vector, 
        limit=limit
    )
    return search_result
