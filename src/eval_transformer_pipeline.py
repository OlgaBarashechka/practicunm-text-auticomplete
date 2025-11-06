from transformers import pipeline
from rouge_score import rouge_scorer
from tqdm import tqdm

def evaluate_transformer(dataloader, tokenizer, device='cpu', max_examples=10):
    """
    Простая и быстрая оценка distilgpt2 на первых max_examples из dataloader.
    """
    # 1. Загружаем модель
    generator = pipeline(
        "text-generation",
        model="distilgpt2",
        device=0 if device == "cuda" else -1,
        torch_dtype="auto",
        pad_token_id=50257,
        return_full_text=False  # только новая часть
    )

    # 2. Собираем промпты и истинные тексты
    prompts = []
    true_texts = []

    for batch in dataloader:
        input_ids = batch['input_ids']  # Изменили доступ к полю
        for i in range(input_ids.size(0)):
            if len(prompts) >= max_examples:
                break
            half = input_ids.size(1) // 2
            if half < 2:
                continue

            # Декодируем промпт и полный текст
            prompt = tokenizer.decode(input_ids[i, :half].tolist(), skip_special_tokens=True)
            full = tokenizer.decode(input_ids[i].tolist(), skip_special_tokens=True)

            prompts.append(prompt)
            true_texts.append(full)

        if len(prompts) >= max_examples:
            break

    if not prompts:  # Исправили опечатку: было 'prompts'
        print("Нет данных.")
        return 0.0, 0.0

    # 3. Генерация и оценка
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    total_r1 = 0.0
    total_r2 = 0.0

    for prompt, true in tqdm(list(zip(prompts, true_texts)), desc="DistilGPT-2"):
        try:
            outputs = generator(prompt, max_new_tokens=10, do_sample=False)
            generated = outputs[0] if isinstance(outputs[0], str) else outputs[0]['generated_text']
            full_generated = prompt + " " + generated.strip()

            scores = scorer.score(true, full_generated)
            total_r1 += scores['rouge1'].fmeasure
            total_r2 += scores['rouge2'].fmeasure
        except Exception as e:
            print(f"Ошибка при обработке промпта: {e}")
            continue  # пропускаем при ошибках

    # 4. Вывод
    count = len(prompts)
    avg_r1 = total_r1 / count
    avg_r2 = total_r2 / count

    print(f"DistilGPT-2 (на {count} примерах):")
    print(f"  ROUGE-1: {avg_r1:.4f}")
    print(f"  ROUGE-2: {avg_r2:.4f}")

    return avg_r1, avg_r2