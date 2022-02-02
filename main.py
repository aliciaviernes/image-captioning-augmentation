from synonym_replacement import eda
from backtranslate import backtranslation
from t5_paraphrase import t5_paraphrase, t5_batchwise_paraphrase
from nltk.tokenize import RegexpTokenizer
import time


def tokenize(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(sentence.lower())


def text_augmentation(sentence):
    # returns 10 augmentations: 2 from backtranslation, 3 from synonym replacement,
    # 5 from T5 paraphrasing.

    backtrans_1 = backtranslation(sentence, [0, 1])
    backtrans_2 = backtranslation(sentence, [0, 2])
    from_eda = eda(sentence)
    paraphrases = t5_paraphrase(sentence)
    
    augmentations = {backtrans_1, backtrans_2}.union(from_eda)
    augmentations = augmentations.union(paraphrases)
    
    return augmentations


def text_augmentation2(sent_batch):

    paraphrases = t5_batchwise_paraphrase(sent_batch)

    for sentence in sent_batch:
        bt1 = backtranslation(sentence, [0, 1])
        bt2 = backtranslation(sentence, [0, 2])
        sr = eda(sentence)
        paraphrases.extend((bt1, bt2, sr))

    return paraphrases


def captions_augment(captions):  # function for show attend and tell
    augmented_captions = list()
    if len(captions) != 5:
        pass
        # print(len(captions))
    for caption in captions: 
        caption = ' '.join(caption)
        augmentations = text_augmentation(caption)
        augmented_captions.extend(augmentations)
    return augmented_captions


if __name__ == "__main__":
    
    batch_sentences = ['this is one sentence', 'and this serves as a second sentence', 'the third sentence is here']

    start_time = time.time()
    augmented_batch = list()
    for caption in batch_sentences:
      augmentations = text_augmentation(caption)
      augmented_batch.extend(augmentations)

    print("--- %s seconds for single sentence augmentation ---" % (time.time() - start_time))
    print(len(augmented_batch))
    del augmented_batch

    start_time = time.time()
    
    # batch_sentences = ['this is one sentence', 'and this serves as a second sentence', 'the third sentence is here']
    augmented_batch = text_augmentation2(batch_sentences)

    print("--- %s seconds for batchwise augmentation ---" % (time.time() - start_time))
    print(len(augmented_batch))

