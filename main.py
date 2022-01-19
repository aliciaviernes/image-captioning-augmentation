from synonym_replacement import eda
from backtranslate import backtranslation
from contextual_t5 import *


def text_augmentation(sentence):
    backtrans_1 = backtranslation(sentence, [0, 1])
    backtrans_2 = backtranslation(sentence, [0, 2])
    from_eda = eda(sentence)
    
    augmentations = {backtrans_1, backtrans_2}.union(from_eda)
    
    return augmentations


def text_augmentation_batch(batch):
    # includes T5 augmentation which happens batchwise 
    # in this case batch == 5 sentences.
    augmentations = set()
    for sentence in batch:
        augs = text_augmentation(sentence)
        augmentations = augmentations.union(augs)
    



if __name__ == "__main__":
    ground_truth = "A man with a red shirt is napping under a tree and children are playing"
    augs = text_augmentation(ground_truth)
    print(len(augs))
