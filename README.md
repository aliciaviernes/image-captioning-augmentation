# image-captioning-augmentation
Augmentation methods for Image Captioning: Text, image, and joint.

## Text augmentation

### Synonym Replacement

Based on EDA synonym replacement (with the help of WordNet).

### Back-translation

Translation from and to Spanish & Arabic. Needed model: argostranslate

### T5: Paraphasing

T5-powered paraphrasing with a finetuned model from Huggingface: `ramsrigouthamg/t5_paraphraser`

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
