from datasets import Dataset
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, pipeline
import torch

TRAINING = True
MAX_LEN = 10
def prep_data():
    data = [
        {
            "text": "The battery life of this phone is great.",
            "aspect_terms": ["battery life"],
            "aspect_terms_idx": [(1, 2)]
        },
        {
            "text": "I love the screen quality and the camera.",
            "aspect_terms": ["screen quality", "camera"],
            "aspect_terms_idx" :[(3,4),(7,7)]
        },
    ]
    with open('./data/train_triplets.txt', 'r') as f:
        text = f.readlines()

    for line in text:
        # Split the line into the text and aspect terms
        line_parts = line.strip().split('####[')
        text = line_parts[0].strip()
        aspect_terms = eval('[' + line_parts[1].strip()[:-1] + ']')
        example = {'text': text, 'aspect_terms': [], 'polarity': [], 'aspect_terms_idx': []}
        for term_indices, polarity_indices, polarity in aspect_terms:
            term_start = term_indices[0]
            term_end = term_indices[1] if len(term_indices) >= 2 else term_indices[0]
            term = text.split(" ")[term_start:term_end+1]
            if len(term) > 1:
                term = " ".join(term)
            else:
                term=term[0]
            example['aspect_terms'].append(term)
            example['aspect_terms_idx'].append((term_start, term_end))
            example['polarity'].append(polarity)
        # Add the example to the appropriate set
        data.append(example)
    ######
    dataset = Dataset.from_dict({'text': [d['text'] for d in data[:2]], 'aspect_terms': [d['aspect_terms'] for d in data[:2]], 'aspect_terms_idx': [d['aspect_terms_idx'] for d in data[:2]]})
    return dataset


def tokenize_and_extract_labels(example):
    full_text = example['text']
    tokens = tokenizer(full_text, return_offsets_mapping=True, is_split_into_words=False, padding='max_length',max_length=MAX_LEN, truncation=True)
    iob_labels = [0 for i in range(MAX_LEN)]

    for start, end in example['aspect_terms_idx']:
        if start < MAX_LEN and end < MAX_LEN:
            iob_labels[start:end + 1] = [2 for _ in range(end+1-start)]
            iob_labels[start] = 1
    
    tokens['labels'] = iob_labels 
    return tokens

def extract_aspect_terms_labels(text, model, tokenizer):
    input_tokens = tokenizer(text, return_tensors="pt")
    output = model(**input_tokens)
    predictions = torch.argmax(output.logits, dim=-1).squeeze().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_tokens['input_ids'].squeeze().numpy())
    aspect_terms = []
    term = ""
    for idx, token in enumerate(tokens):
        label = predictions[idx]
        print(label)
        if label == 1:  # 'B-ASP'
            if term:
                aspect_terms.append(term)
            term = token
        elif label == 2:  # 'I-ASP'
            term += " " + token
        else:  # 'O'
            if term:
                aspect_terms.append(term)
                term = ""

    if term:
        aspect_terms.append(term)
    return aspect_terms

if TRAINING:
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = prep_data()
    dataset = dataset.map(tokenize_and_extract_labels)
    dataset =  dataset.remove_columns(['aspect_terms','offset_mapping','aspect_terms_idx'])
    train_dataset = dataset
    dataset = dataset.train_test_split(test_size=0.5)
    test_dataset = dataset['test']
    # train_dataset = test_dataset = dataset
    print(dataset['train'][0])
    print(dataset['test'][0])

    test_dataset.remove_columns(['labels'])
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=3)

    training_args = TrainingArguments(
        output_dir="./output",
        overwrite_output_dir=True,
        num_train_epochs=20,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=100,
        eval_steps=1000,
        evaluation_strategy="steps",
        learning_rate=1e-2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    # model.save_pretrained("./models/aspect_term_extraction_model")
    # tokenizer.save_pretrained("./models/aspect_term_extraction_model")
    # # Test the extraction function
    # text = "The battery life of this phone is great."
    # aspect_terms = extract_aspect_terms_labels(text, model, tokenizer)
    # print("Aspect terms:", aspect_terms)
    predictions = trainer.predict(test_dataset)
    print(predictions.label_ids)
else:
    # Load the fine-tuned model and tokenizer
    model_path = "./aspect_term_extraction_model"
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Test the extraction function
    text = "The car is good."
    aspect_terms = extract_aspect_terms_labels(text, model, tokenizer)
    print("Aspect terms:", aspect_terms)