import os
import random
from nltk.corpus import wordnet

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ")
            if synonym != word:
                synonyms.add(synonym)
    return list(synonyms)

def synonym_replacement(sentence):
    words = sentence.split()
    new_sentence = words.copy()
    for i, word in enumerate(words):
        synonyms = get_synonyms(word)
        if synonyms:
            new_sentence[i] = random.choice(synonyms)
    return ' '.join(new_sentence)

def random_insertion(sentence):
    words = sentence.split()
    new_sentence = words.copy()
    for _ in range(2):
        synonyms = []
        while not synonyms:
            random_word = random.choice(words)
            synonyms = get_synonyms(random_word)
        random_synonym = random.choice(synonyms)
        random_idx = random.randint(0, len(new_sentence))
        new_sentence.insert(random_idx, random_synonym)
    return ' '.join(new_sentence)

def random_deletion(sentence, p=0.2):
    words = sentence.split()
    if len(words) == 1:
        return sentence
    new_sentence = [word for word in words if random.uniform(0, 1) > p]
    if not new_sentence:
        return random.choice(words)
    return ' '.join(new_sentence)

def random_swap(sentence):
    words = sentence.split()
    new_sentence = words.copy()
    idx1, idx2 = random.sample(range(len(words)), 2)
    new_sentence[idx1], new_sentence[idx2] = new_sentence[idx2], new_sentence[idx1]
    return ' '.join(new_sentence)

def augment_text_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    augmented_lines = []
    for line in lines:
        line = line.strip()
        if line:
            augmented_lines.append(line)
            augmented_lines.append(synonym_replacement(line))
            augmented_lines.append(random_insertion(line))
            augmented_lines.append(random_deletion(line))
            augmented_lines.append(random_swap(line))
    
    return augmented_lines

def save_augmented_dataset(original_path, augmented_lines):
    base_name = os.path.basename(original_path)
    dir_name = os.path.dirname(original_path)
    new_file_path = os.path.join(dir_name, f"augmented_{base_name}")
    with open(new_file_path, 'w') as new_file:
        for line in augmented_lines:
            new_file.write(line + '\n')

def augment_dataset(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            augmented_lines = augment_text_file(file_path)
            save_augmented_dataset(file_path, augmented_lines)

# Folder containing text files to be augmented
dataset_folder = 'C:\\Personal\\MS Program\\Courses\\CSC525-1\\Module 5\\dataset\\'
augment_dataset(dataset_folder)
