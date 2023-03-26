import ast
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer
from transformers import BertForTokenClassification
import numpy as np
from transformers import AdamW


class ASTETokenizer(BertTokenizer):     
    def subword_tokenize(self, tokens, labels):
        split_tokens, split_labels= [], []
        idx_map=[]
        for ix, token in enumerate(tokens):
            sub_tokens=self.wordpiece_tokenizer.tokenize(token)
            for jx, sub_token in enumerate(sub_tokens):
                split_tokens.append(sub_token)
                if labels[ix]=="B" and jx>0:
                    split_labels.append("I")
                else:
                    split_labels.append(labels[ix])
                idx_map.append(ix)
        return split_tokens, split_labels, idx_map

class ASTE_Dataset(Dataset):
    def __init__(self, path) -> None:
        self.path = path
        self.X, self.Y = self.read_data()

    def encode_label(self, label):
        if label == 'POS':
            return torch.eye(3)[0]
        elif label == 'NEU':
            return torch.eye(3)[1]
        else:
            return torch.eye(3)[2]

    def read_data(self):
        X, Y = [], []

        with open(self.path) as f:
            lines = f.readlines()
            for line in lines[:10]:
                phrase, annotation = line.split('####')
                annotation = ast.literal_eval(annotation)[0]

                # words = phrase.split()
                # target = ' '.join([words[i] for i in annotation[0]])
                # opinion = ' '.join([words[i] for i in annotation[1]])
                # sentiment = self.encode_label(annotation[2])
                target, opinion, sentiment = annotation

                X.append(phrase)
                Y.append([target, opinion, self.encode_label(sentiment)])

        return X, Y

if __name__ == '__main__':
    dataset = ASTE_Dataset('./14lap/train_triplets.txt')
    # tokenizer = ASTETokenizer.from_pretrained('./models/BERT.json')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # input_text = '[CLS] ' + phrase + ' [SEP] ' + target + ' [SEP] ' + opinion + ' [SEP]'

    # input_ids = tokenizer.encode(input_text, add_special_tokens=True)
    # print(input_ids)


    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    # outputs = model(torch.Tensor(input_ids))
    # predicted_label = outputs.logits.argmax().item()
    # print(predicted_label)

    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    phrase = dataset.X[0]
    target, opinion, sentiment = dataset.Y[0]
    
    tagger = lambda i, data: 1 if i == data[0] else 2 
    train_aspect_labels  = [0 if i not in target else tagger(i, target) for i in range(len(phrase.split()))]
    train_opinion_labels  = [0 if i not in opinion else tagger(i, opinion) for i in range(len(phrase.split()))]
    
    tokenized_texts = [tokenizer.tokenize(sent) for sent in [phrase]]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    input_ids = torch.tensor(input_ids)
    train_aspect_labels = torch.tensor(train_aspect_labels)
    train_opinion_labels = torch.tensor(train_opinion_labels)

    batch_size = 19
    epochs = 3
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    # Train the model
    model.train()

    for epoch in range(epochs):
        for i in range(0, len(input_ids), batch_size):
            inputs = input_ids[i:i+batch_size]
            aspect_labels = train_aspect_labels[i:i+batch_size]
            opinion_labels = train_opinion_labels[i:i+batch_size]
            outputs = cde(inputs, labels=aspect_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    predicted_label = torch.argmax(outputs.logits, dim = 2)
    print(predicted_label)
