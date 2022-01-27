import albumentations as A
import cv2

"""
# Declare an augmentation pipeline
transform = A.Compose([
    # A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.7),
    A.RandomBrightnessContrast(p=0.7),
    
])
"""

def image_transform(imgpath, save=True):
    transform = A.Compose([
        A.CLAHE(),
        A.RandomRotate90(),
        A.Transpose(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
        A.Blur(blur_limit=3),
        A.OpticalDistortion(),
        A.GridDistortion(),
        A.HueSaturationValue(),
        A.HorizontalFlip(p=0.7),
    ])

    image = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB) 
    aug_image = cv2.cvtColor(transform(image=image)['image'], cv2.COLOR_RGB2BGR)
    if save == True:
        cv2.imwrite(imgpath.replace('.jpg', '_aug.jpg'), aug_image)


if __name__ == "__main__":
    image_transform('./image.jpg')