import pandas as pd
import re
import string
import os
import requests
from io import StringIO
from sklearn.model_selection import train_test_split

def download_file(url, output_path):
    """Скачивает файл по URL и сохраняет в указанную папку."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Файл успешно загружен: {output_path}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Ошибка загрузки файла: {e}")

def load_and_clean_data(filepath_or_url, local_path="data/raw_dataset.txt"):
    # Если файл не существует локально — скачиваем
    if not os.path.exists(local_path):
        print("Файл не найден локально. Начинаем загрузку...")
        download_file(filepath_or_url, local_path)
    
    # Читаем текстовый файл (построчно)
    with open(local_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Создаём DataFrame, сразу фильтруя пустые строки
    df = pd.DataFrame({'text': [line.strip() for line in lines if line.strip()]})
    
    # Очистка текста 
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Применяем очистку и **сразу фильтруем пустые результаты**
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.strip() != '']  # Удаляем строки, ставшие пустыми после очистки
    
    return df

def split_dataset(df, train_ratio=0.8, val_ratio=0.1):
    # Убедимся, что в df нет пустых строк (дополнительная проверка)
    df = df[df['text'].str.strip() != ''].reset_index(drop=True)
    
    train_df = df.sample(frac=train_ratio, random_state=42)
    remaining = df.drop(train_df.index)
    val_df = remaining.sample(frac=val_ratio/(1-train_ratio), random_state=42)
    test_df = remaining.drop(val_df.index)
    
    # Финальная проверка на пустые строки в каждом подмножестве
    train_df = train_df[train_df['text'].str.strip() != ''].reset_index(drop=True)
    val_df = val_df[val_df['text'].str.strip() != ''].reset_index(drop=True)
    test_df = test_df[test_df['text'].str.strip() != ''].reset_index(drop=True)
    
    return train_df, val_df, test_df

def save_datasets(train_df, val_df, test_df, output_dir):
    # Перед записью — финальная проверка на пустоту
    for df, name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
        df_clean = df[df['text'].str.strip() != ''].reset_index(drop=True)
        filepath = f"{output_dir}/{name}.csv"
        df_clean.to_csv(filepath, index=False)
        print(f"Файл успешно записан: {filepath}")
    print(f"train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")