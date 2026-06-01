import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class RestorationDataset(Dataset):
    def __init__(
        self,
        degraded_dir,
        clean_dir=None,
        image_size=256,
        augment=False,
        crop_size=None,
    ):
        self.degraded_dir = degraded_dir
        self.clean_dir = clean_dir
        self.image_size = image_size
        self.augment = augment
        self.crop_size = crop_size

        self.files = sorted([
            f for f in os.listdir(degraded_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self):
        return len(self.files)

    def get_clean_name(self, degraded_name):
        if degraded_name.startswith("rain-"):
            number = degraded_name.replace("rain-", "").replace(".png", "")
            return f"rain_clean-{number}.png"
        if degraded_name.startswith("snow-"):
            number = degraded_name.replace("snow-", "").replace(".png", "")
            return f"snow_clean-{number}.png"
        raise ValueError(f"Unknown degraded image name: {degraded_name}")

    def get_degradation_label(self, degraded_name):
        """0 = rain, 1 = snow, -1 = unknown."""
        if degraded_name.startswith("rain-"):
            return 0
        if degraded_name.startswith("snow-"):
            return 1
        return -1

    def paired_random_crop(self, degraded_image, clean_image):
        width, height = degraded_image.size
        crop_size = self.crop_size

        if width < crop_size or height < crop_size:
            degraded_image = TF.resize(
                degraded_image, [self.image_size, self.image_size]
            )
            clean_image = TF.resize(
                clean_image, [self.image_size, self.image_size]
            )
            width, height = degraded_image.size

        left = random.randint(0, width - crop_size)
        top = random.randint(0, height - crop_size)
        degraded_image = TF.crop(degraded_image, top, left, crop_size, crop_size)
        clean_image = TF.crop(clean_image, top, left, crop_size, crop_size)
        return degraded_image, clean_image

    def apply_augmentation(self, degraded_image, clean_image, label):
        # ---- Spatial augmentation ----
        if self.crop_size is not None:
            degraded_image, clean_image = self.paired_random_crop(
                degraded_image, clean_image
            )
        else:
            degraded_image = TF.resize(
                degraded_image, [self.image_size, self.image_size]
            )
            clean_image = TF.resize(
                clean_image, [self.image_size, self.image_size]
            )

        # Horizontal flip: safe for both rain and snow
        if random.random() < 0.5:
            degraded_image = TF.hflip(degraded_image)
            clean_image = TF.hflip(clean_image)

        if label == 1:
            # Snow: direction-independent — full augmentation
            if random.random() < 0.5:
                degraded_image = TF.vflip(degraded_image)
                clean_image = TF.vflip(clean_image)
            angle = random.choice([0, 90, 180, 270])
            degraded_image = TF.rotate(degraded_image, angle)
            clean_image = TF.rotate(clean_image, angle)
        else:
            # Rain: directional — only 180° rotation is safe
            angle = random.choice([0, 180])
            degraded_image = TF.rotate(degraded_image, angle)
            clean_image = TF.rotate(clean_image, angle)

        # ---- Brightness augmentation (KEY FIX) ----
        # Apply the SAME brightness factor to both degraded and clean.
        # This simulates images taken under different lighting conditions,
        # including the bright snow scenes that dominate the test set.
        # Without this, the model never sees inputs with mean > 200 during
        # training and catastrophically over-darkens bright test images.
        if random.random() < 0.4:
            # Wide range: 0.6x (dark) to 1.5x (very bright)
            factor = random.uniform(0.6, 1.5)
            degraded_image = TF.adjust_brightness(degraded_image, factor)
            clean_image = TF.adjust_brightness(clean_image, factor)

        return degraded_image, clean_image

    def __getitem__(self, index):
        degraded_name = self.files[index]
        degraded_path = os.path.join(self.degraded_dir, degraded_name)
        degraded_image = Image.open(degraded_path).convert("RGB")

        # Inference mode
        if self.clean_dir is None:
            degraded_image = TF.resize(
                degraded_image, [self.image_size, self.image_size]
            )
            return TF.to_tensor(degraded_image), degraded_name

        # Training / validation mode
        label = self.get_degradation_label(degraded_name)
        clean_name = self.get_clean_name(degraded_name)
        clean_path = os.path.join(self.clean_dir, clean_name)
        clean_image = Image.open(clean_path).convert("RGB")

        if self.augment:
            degraded_image, clean_image = self.apply_augmentation(
                degraded_image, clean_image, label
            )
        else:
            degraded_image = TF.resize(
                degraded_image, [self.image_size, self.image_size]
            )
            clean_image = TF.resize(
                clean_image, [self.image_size, self.image_size]
            )

        return TF.to_tensor(degraded_image), TF.to_tensor(clean_image), label