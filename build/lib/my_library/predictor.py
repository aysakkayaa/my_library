from my_library.model_handling import load_model, encode_texts, load_data, prepare_data
from my_library.qdrant_operations import connect_qdrant, create_collection, upload_vectors, search_collection
from my_library.data_preprocessing import preprocess_text, get_detailed_instruct

class TextPredictor:
    def __init__(self, qdrant_url: str, collection_name: str):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.model = load_model()  # Modeli yükle
        self.client = connect_qdrant(host=qdrant_url)  # Qdrant'a bağlan
        self.train_df = None
        self.test_df = None

    def load_and_prepare_data(self, file_path: str):
        """Veriyi yükle ve train-test setlerine ayır."""
        df = load_data(file_path)
        self.train_df, self.test_df = prepare_data(df)
        self._train_model()

    def _train_model(self):
        """Train verisi ile modeli eğit ve Qdrant'a yükle."""
        if self.train_df is not None:
            task_description = "Text Classification"
            train_texts = self.train_df['text'].tolist()
            formatted_texts = [get_detailed_instruct(task_description, text) for text in train_texts]
            train_embeddings = encode_texts(self.model, formatted_texts)

            # Koleksiyonun mevcut olup olmadığını kontrol edin ve varsa silin
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)

            # Yeni koleksiyon oluşturun
            create_collection(self.client, self.collection_name, vector_size=train_embeddings.shape[1])

            # Vektörleri ve etiketleri Qdrant'a batch olarak yükleyin
            train_labels = self.train_df[['level_0', 'level_1', 'level_2', 'level_3', 'level_4', 'label']].values.tolist()
            upload_vectors(self.client, self.collection_name, train_embeddings, train_texts, train_labels, batch_size=100)
        else:
            raise ValueError("Train data is not loaded.")
        
        

    def predict_test_set(self, limit: int = 3):
        """Test setini kullanarak her bir test verisi için en benzer cümleleri bul."""
        if self.test_df is not None:
            results = []
            task_description = "Text Classification"

            for i, row in self.test_df.iterrows():
                test_text = row['text']
                preprocessed_text = preprocess_text(test_text)
                formatted_text = get_detailed_instruct(task_description, preprocessed_text)
                query_embedding = encode_texts(self.model, [formatted_text])[0]

                # En yakın sonuçları Qdrant'tan al
                search_results = search_collection(self.client, self.collection_name, query_embedding, limit=limit)

                # Sonuçları sakla
                results.append({
                    'input_text': test_text,
                    'matches': [
                        {
                            'text': match.payload['text'],
                            'level_0': match.payload['level_0'],
                            'level_1': match.payload['level_1'],
                            'level_2': match.payload['level_2'],
                            'level_3': match.payload['level_3'],
                            'level_4': match.payload['level_4'],
                            'label': match.payload['label']
                        }
                        for match in search_results
                    ]
                })

            return results
        else:
            raise ValueError("Test data is not loaded.")
