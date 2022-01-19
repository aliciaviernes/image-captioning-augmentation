from synonym_replacement import eda
from backtranslate import backtranslation
from t5_paraphrase import t5_paraphrase
from nltk.tokenize import RegexpTokenizer


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


def tokenize(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(sentence.lower())


if __name__ == "__main__":
    ground_truth = "A man with a red shirt is napping under a tree and children are playing"
    augs = text_augmentation(ground_truth)
    print('\n' + ground_truth + '\n')
    for i, aug in enumerate(augs):
        print(f"{i}: {tokenize(aug)}")
