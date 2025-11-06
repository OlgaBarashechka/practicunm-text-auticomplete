
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.lstm_model import LSTMModel
from src.next_token_dataset import NextTokenDataset
import numpy as np

def train_model(train_loader, val_loader, vocab_size, epochs=10, lr=0.001, device='cpu'):
    model = LSTMModel(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # 1. Получаем вывод модели (кортеж: логиты, hidden)
            output = model(input_ids)
            
            # 2. Извлекаем логиты (первый элемент кортежа)
            logits = output[0]  # <-- Ключевая строка!

            # 3. Проверяем тип и размерность
            if not isinstance(logits, torch.Tensor):
                raise TypeError(f"logits не является тензором: {type(logits)}")
            if logits.dim() != 3:
                raise ValueError(f"logits имеет {logits.dim()} измерений, ожидается 3: {logits.shape}")

            # 4. Вычисляем потерю (используем ТОЛЬКО logits, а не output!)
            loss = criterion(
                logits.view(-1, vocab_size),  # Теперь это тензор, а не кортеж
                labels.view(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
       
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')
        scheduler.step()

    return model

