import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        """
        Forward pass модели.
        
        Args:
            x: входные индексы токенов (batch_size, seq_len)
            hidden: начальное скрытое состояние (h0, c0) или None
        
        Returns:
            logits: логиты для классификации (batch_size, seq_len, vocab_size)
            hidden: новое скрытое состояние (h_n, c_n)
        """
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        logits = self.fc(x)  # Назвали явно logits для понятности
        return logits, hidden

    def init_hidden(self, batch_size, device='cpu'):
        """Инициализация скрытого состояния."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

    def generate(self, tokenizer, prompt, max_length=20, temperature=1.0, device='cpu'):
        """
        Генерация текста по промту.
        
        Args:
            tokenizer: токенизатор
            prompt: входной текст-промт
            max_length: максимальная длина генерируемого текста
            temperature: температура для разнообразия генерации
            device: устройство ('cpu' или 'cuda')
        
        Returns:
            Сгенерированный текст
        """
        self.eval()
        with torch.no_grad():
            # Токенизация промта
            tokens = tokenizer.encode(prompt.lower(), return_tensors='pt').to(device)
            generated = tokens.clone()
            
            # Инициализация скрытого состояния
            batch_size = tokens.size(0)
            hidden = self.init_hidden(batch_size, device)

            for _ in range(max_length - tokens.size(1)):
                # Forward pass с текущим скрытым состоянием
                logits, hidden = self.forward(generated, hidden)
                
                # Берём только последний токен для предсказания следующего
                next_token_logits = logits[:, -1, :] / temperature
                
                # Применяем softmax и выбираем следующий токен
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Добавляем новый токен к сгенерированной последовательности
                generated = torch.cat([generated, next_token], dim=1)

                # Проверяем EOS-токен
                if next_token.item() == tokenizer.eos_token_id:
                    break

            # Декодируем в текст
            text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
            return text