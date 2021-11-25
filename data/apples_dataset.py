from data.base_dataset import BaseDataset
import os
import numpy as np
import albumentations as A
from torchvision.transforms import ToTensor


class ApplesDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.root = opt.dataroot
        rgb_folder, msi_folder = os.path.join(self.root, 'rgb'), os.path.join(self.root, 'msi')
        rgb_paths = sorted(list(map(lambda x: os.path.join(rgb_folder, x), os.listdir(rgb_folder))))
        msi_paths = sorted(list(map(lambda x: os.path.join(msi_folder, x), os.listdir(msi_folder))))
        self.rgb_paths = rgb_paths
        self.msi_paths = msi_paths
        self.image_paths = [*rgb_paths, *msi_paths]
        transform = A.Compose([
            # TODO: or (256, 320)?
            A.RandomCrop(256, 256, p=1.0),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.4, scale_limit=0.3, rotate_limit=90, p=0.5),
        ], additional_targets={'other_image': 'image'}
        )

        self.transform = None if opt.phase == 'test' else transform
        # self.resize = A.Compose([A.Resize(opt.size, opt.size)], additional_targets={'other_image': 'image'}) \
        #     if opt.size > 0 else None
        self.to_tensor = ToTensor()
        self.nir_channels_only = opt.nir_channels_only

    def __getitem__(self, index):
        path_rgb = self.rgb_paths[index]
        path_msi = self.msi_paths[index]
        rgb = np.load(path_rgb)
        msi = np.load(path_msi)

        if self.nir_channels_only:
            msi = msi[:, :, [5, 6, 7]]

        if self.transform is not None:
            transformed = self.transform(image=rgb, other_image=msi)
            rgb, msi = transformed["image"], transformed["other_image"]

        # if self.resize is not None:
        #     resized = self.resize(image=rgb, other_image=msi)
        #     rgb, msi = resized["image"], resized["other_image"]

        return {'A': self.to_tensor(rgb), 'B': self.to_tensor(msi), 'A_paths': path_rgb, 'B_paths': path_msi}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths) // 2
