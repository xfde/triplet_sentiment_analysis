import json

# Read in the text file
with open('./data/train_triplets.txt', 'r') as f:
    text = f.readlines()

# Create an empty dictionary to hold the dataset
dataset = {'train': [], 'validation': []}

# Loop over each line in the text file
for line in text:
    # Split the line into the text and aspect terms
    line_parts = line.strip().split('####[')
    text = line_parts[0].strip()
    aspect_terms = eval('[' + line_parts[1].strip()[:-1] + ']')
    
    # Determine which set to add the example to (train or validation)
    if len(dataset['train']) < len(dataset['validation']):
        set_key = 'train'
    else:
        set_key = 'validation'
    
    # Create a dictionary for the example
    example = {'text': text, 'aspect_terms': []}
    
    # Loop over each aspect term and add it to the example
    for term_indices, polarity_indices, polarity in aspect_terms:
        term_start = term_indices[0]
        term_end = term_indices[1] if len(term_indices) >= 2 else term_indices[0]
        term = text[term_start:term_end]
        example['aspect_terms'].append({'term': term, 'polarity': polarity})
    
    # Add the example to the appropriate set
    dataset[set_key].append(example)

# Write the dataset to a JSON file
with open('semeval_2014_dataset.json', 'w') as f:
    json.dump(dataset, f)