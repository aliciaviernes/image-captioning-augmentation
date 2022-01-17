from synonym_replacement import eda
from backtranslate import backtranslation


augmentations = set()

backtranslated_1 = backtranslation(ground_truth, [0, 1])  # from Arabic
backtranslated_2 = backtranslation(ground_truth, [0, 2])  # from Spanish
fromEDA = eda(ground_truth)

augmentations.add(backtranslated_1)
augmentations.add(backtranslated_2)
augmentations = augmentations.union(fromEDA)
print(augmentations)

def text_augmentation(sentence):
    backtrans_1 = backtranslation(sentence, [0, 1])
    backtrans_2 = backtranslation(sentence, [0, 2])
    from_eda = eda(sentence)
    
    augmentations = {backtrans_1, backtrans_2}.union(from_eda)
    
    return augmentations


if __name__ == "__main__":
    ground_truth = "A man with a red shirt is napping under a tree and children are playing"
    augs = text_augmentation(ground_truth)
