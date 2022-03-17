from synonym_replacement import eda
from backtranslate import backtranslation
from t5_paraphrase import t5_batchwise_paraphrase
from nltk.tokenize import RegexpTokenizer
import time


def tokenize(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(sentence.lower())


def text_augmentation(sent_batch):

    paraphrases = t5_batchwise_paraphrase(sent_batch)

    for sentence in sent_batch:
        bt1 = backtranslation(sentence, [0, 1])
        bt2 = backtranslation(sentence, [0, 2])
        sr = eda(sentence)
        paraphrases.extend((bt1, bt2))
        paraphrases.extend(list(sr))

    return paraphrases


def text_augmentation_annotated(sent_batch):

    paraphrases = t5_batchwise_paraphrase(sent_batch)
    for i in range(len(paraphrases)):
        paraphrases[i] = 'Paraphrase: ' + paraphrases[i]

    for sentence in sent_batch:
        bt1 = 'Backtranslation 1: ' + backtranslation(sentence, [0, 1])
        bt2 = 'Backtranslation 2: ' + backtranslation(sentence, [0, 2])
        sr = eda(sentence)
        paraphrases.extend((bt1, bt2))
        for aug in sr:
            paraphrases.append('Synonym replacement: ' + aug)
    
    return paraphrases


def captions_augment(captions):  # function for show attend and tell
    for i in range(len(captions)): 
        captions[i] = ' '.join(captions[i])
    return text_augmentation(captions)


if __name__ == "__main__":
    
    captions = [
        ['a', 'woman', 'wearing', 'a', 'net', 'on', 'her', 'head', 'cutting', 'a', 'cake'], 
        ['a', 'woman', 'cutting', 'a', 'large', 'white', 'sheet', 'cake'], 
        ['a', 'woman', 'wearing', 'a', 'hair', 'net', 'cutting', 'a', 'large', 'sheet', 'cake'], 
        ['there', 'is', 'a', 'woman', 'that', 'is', 'cutting', 'a', 'white', 'cake'], 
        ['a', 'woman', 'marking', 'a', 'cake', 'with', 'the', 'back', 'of', 'a', 'chefs', 'knife']
        ]

    start_time = time.time()
    print("--- %s seconds for something ---" % (time.time() - start_time))
