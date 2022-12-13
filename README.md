# Image Captioning Augmentation

Augmentation methods for Image Captioning: Text, image, and joint.

## Text augmentation

### Synonym Replacement

Based on EDA synonym replacement (with the help of [WordNet](https://wordnet.princeton.edu/)).

### Back-translation

Translation from and to Spanish & Arabic. Needed model: [argostranslate](https://github.com/argosopentech/argos-translate)

1. Create a directory `argostranslate`.
2. Download the models specified in `backtranslate.py`.

### Pegasus Paraphrase

T5-powered paraphrasing with a finetuned model from Huggingface: `tuner007/pegasus_paraphrase`

## Image augmentation

`Albumentations` is leveraged for image augmentation. The pipeline contains the following transformations:

1. CLAHE
2. RandomRotate90
3. Transpose
4. ShiftScaleRotate
5. Blur
6. OpticalDistortion
7. GridDistortion
8. HueSaturationValue
9. HorizontalFlip.
