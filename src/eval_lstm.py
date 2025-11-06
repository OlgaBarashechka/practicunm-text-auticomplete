from rouge_score import rouge_scorer
from tqdm import tqdm
import torch

def evaluate_lstm(model, dataloader, tokenizer, device='cpu'):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    total_rouge1 = 0
    total_rouge2 = 0
    count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating LSTM"):
            # Проверяем тип данных
            if isinstance(batch, dict):
                input_ids = batch['input_ids']
            else:
                input_ids = batch  # если batch — это просто тензор

            # Убеждаемся, что input_ids — тензор
            if not isinstance(input_ids, torch.Tensor):
                raise TypeError(f"input_ids должен быть тензором, получен {type(input_ids)}")

            input_ids = input_ids.to(device)

            for i in range(input_ids.size(0)):
                # Делим последовательность пополам
                seq_len = input_ids.size(1)
                prompt_ids = input_ids[i, :seq_len // 2]
                target_ids = input_ids[i, seq_len // 2:]

                # Декодируем промпт и истинный текст
                prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
                true_text = tokenizer.decode(target_ids, skip_special_tokens=True)

                # Генерируем текст
                generated = model.generate(tokenizer, prompt, max_length=30, device=device)

                # Считаем ROUGE
                scores = scorer.score(true_text, generated)
                total_rouge1 += scores['rouge1'].fmeasure
                total_rouge2 += scores['rouge2'].fmeasure
                count += 1

    if count == 0:
        print("Не было обработано ни одного примера!")
        return 0.0, 0.0

    avg_rouge1 = total_rouge1 / count
    avg_rouge2 = total_rouge2 / count

    print(f"LSTM ROUGE-1: {avg_rouge1:.4f}")
    print(f"LSTM ROUGE-2: {avg_rouge2:.4f}")

    return avg_rouge1, avg_rouge2