import json
import pickle

# Load the uploaded JSON file
json_file_path = './bert_tokenizer_en.json'

# Read the JSON data from the file
with open(json_file_path, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# Extract the "vocab" part from the JSON data
vocab = json_data['model']['vocab']

# Create the corpus in the desired format
corpus = [{str(index): token} for token, index in vocab.items()]

# Output file paths
txt_file_path = './sentences_en_ids_1.txt'
pkl_file_path = './sentences_en_ids_1.pkl'

# Save the corpus to a txt file
with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
    for item in corpus:
        for index, token in item.items():
            txt_file.write(f"{index}: {token}\n")

# Save the corpus to a pickle file
with open(pkl_file_path, 'wb') as pkl_file:
    pickle.dump(corpus, pkl_file)

(txt_file_path, pkl_file_path)  # Output the file paths for the user to access


    