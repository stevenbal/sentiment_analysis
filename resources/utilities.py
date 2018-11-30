import re

def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'(@|http:\/\/)[^\s]*', '', sentence)
    sentence = re.sub(r'[a-z]+\/[a-z]+', ' ', sentence)
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    sentence = ' '.join(sentence.split())
    return sentence