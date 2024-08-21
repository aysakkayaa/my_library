import pandas as pd
from sentence_transformers import SentenceTransformer

def load_model(model_name: str = "intfloat/multilingual-e5-large-instruct"):
    model = SentenceTransformer(model_name)
    return model

def encode_texts(model, texts: list):
    embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    return embeddings

def load_data(file_path: str):
    df = pd.read_excel(file_path)
    return df

def prepare_data(df: pd.DataFrame):
    label_counts = df['label'].value_counts()
    df['count'] = df['label'].map(label_counts)
    filtered_df = df[df['count'] > 1]

    test_df = filtered_df.groupby('label').apply(lambda x: x.sample(2)).reset_index(drop=True)
    train_df = filtered_df[~filtered_df.index.isin(test_df.index)]

    return train_df, test_df
