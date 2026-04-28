import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio
import shutil
import json
from scipy.signal import savgol_filter

# Problem samples in the SCUT-DHGA dataset
ill_list = [
    '1_1_0_2_6',
    '1_1_105_0_2',
    '1_1_108_5_3',
    '1_1_108_5_7',
    '1_1_114_2_5',
    '1_1_116_0_5',
    '1_1_11_0_3',
    '1_1_11_0_9',
    '1_1_128_4_4',
    '1_1_134_2_2',
    '1_1_27_0_5',
    '1_1_39_0_2',
    '1_1_51_0_2',
    '1_1_53_0_2',
    '1_1_67_0_0',
    '1_1_74_0_2',
    '1_1_83_5_4',
    '1_1_93_0_9',
    '1_1_93_1_2',
    '2_1_30_2_0',
    '2_1_34_5_1',
    '2_1_3_3_7',
    '2_1_43_2_3',
    '2_1_47_1_7',
    '2_2_19_0_1',
    '2_2_26_2_4',
    # The samples above have missing keypoints, while the samples below have keypoint coordinates that exceed the image boundary
    '1_1_104_0_0',
    '1_1_104_0_1',
    '1_1_104_0_2',
    '1_1_104_0_4',
    '1_1_104_0_5',
    '1_1_11_0_0',
    '1_1_121_0_5',
    '1_1_121_0_6',
    '1_1_121_0_7',
    '1_1_121_0_8',
    '1_1_121_0_9',
    '1_1_126_0_4',
    '1_1_128_0_4',
    '1_1_134_0_5',
    '1_1_134_0_6',
    '1_1_134_0_7',
    '1_1_134_0_8',
    '1_1_134_0_9',
    '1_1_137_0_8',
    '1_1_138_0_0',
    '1_1_138_0_1',
    '1_1_138_0_3',
    '1_1_139_0_6',
    '1_1_15_0_5',
    '1_1_1_0_5',
    '1_1_27_0_8',
    '1_1_28_0_1',
    '1_1_28_0_4',
    '1_1_32_0_4',
    '1_1_32_0_5',
    '1_1_32_0_6',
    '1_1_32_0_8',
    '1_1_32_0_9',
    '1_1_39_0_3',
    '1_1_39_0_6',
    '1_1_39_0_8',
    '1_1_39_0_9',
    '1_1_40_0_3',
    '1_1_41_0_6',
    '1_1_45_0_2',
    '1_1_45_0_8',
    '1_1_4_0_7',
    '1_1_4_0_8',
    '1_1_54_0_0',
    '1_1_54_0_1',
    '1_1_54_0_2',
    '1_1_54_0_3',
    '1_1_54_0_6',
    '1_1_54_0_7',
    '1_1_54_0_8',
    '1_1_54_0_9',
    '1_1_57_0_0',
    '1_1_57_0_4',
    '1_1_58_0_1',
    '1_1_58_0_5',
    '1_1_58_0_6',
    '1_1_58_0_7',
    '1_1_58_0_8',
    '1_1_67_0_7',
    '1_1_68_0_6',
    '1_1_68_0_7',
    '1_1_68_0_8',
    '1_1_69_0_4',
    '1_1_78_0_4',
    '1_1_78_0_5',
    '1_1_78_0_6',
    '1_1_78_0_7',
    '1_1_78_0_9',
    '1_1_80_0_6',
    '1_1_84_0_8',
    '1_1_86_0_3',
    '1_1_92_0_1',
    '1_1_93_0_4',
    '1_1_94_0_7',
    '1_1_99_0_3',
    '1_1_99_0_5',
    '1_1_99_0_7',
    '1_1_99_0_8',
    '1_1_99_0_9',
    '1_1_9_0_2',
    '1_1_9_0_7',
    '1_1_9_0_9',
    '2_1_15_0_3',
    '2_1_16_0_6',
    '2_1_27_0_5',
    '2_1_30_0_4',
    '2_1_30_0_8',
    '2_1_34_0_9',
    '2_1_40_0_0',
    '2_1_40_0_1',
    '2_1_40_0_2',
    '2_1_40_0_4',
    '2_1_40_0_9',
    '2_1_43_0_9',
    '2_1_47_0_3',
    '2_1_47_0_4',
    '2_1_47_0_5',
    '2_1_47_0_8',
    '2_1_4_0_1',
    '2_1_7_0_1',
    '2_1_7_0_2',
    '2_1_7_0_3',
    '2_2_20_0_6',
    '2_2_26_0_1',
    '2_2_27_0_7',
    '2_2_27_0_8',
    '2_2_28_0_2',
    '2_2_28_0_6',
    '2_2_28_0_7',
    '2_2_40_0_2',
    '2_2_9_0_6',
]
# Problem samples in SCUT-RealDHGA
ill_list_real = [
    '1_1_12_2_2',
]

# Euclidean distance function
euclid_distance = lambda x, y: np.sqrt(np.sum((x - y) ** 2))

def vis_pose(kpt: np.ndarray, img=None, save_name=None):
    '''
    Visualize hand keypoints on an image.

    Args:
        kpt: Keypoint array with shape [21, 2]
        img: Image array (optional)
        save_name: File path to save the visualization (optional)
    '''
    color_lst = ["yellow", "blue", "green", "cyan", "magenta"]
    group_lst = [1, 5, 9, 13, 17, 21]
    plt.figure()
    if img is not None:
        if np.max(img) < 1:
            plt.imshow(img * 255)
        else:
            plt.imshow(img)
    for j, color in enumerate(color_lst):
        x = np.insert(kpt[group_lst[j]:group_lst[j + 1], 0], 0, kpt[0, 0])
        y = np.insert(kpt[group_lst[j]:group_lst[j + 1], 1], 0, kpt[0, 1])
        plt.plot(x, y, color=color, linewidth=3, marker=".", markerfacecolor="red", markersize=15, markeredgecolor="red")
    plt.axis('off')
    plt.xlim((0, 200))
    plt.ylim((200, 0))
    if save_name is not None:
        if not os.path.exists(os.path.dirname(save_name)):
            os.makedirs(os.path.dirname(save_name))
        plt.savefig(save_name, bbox_inches='tight', dpi=300, pad_inches=0)
    plt.show()

class BILASnorm:
    """Background, Illumination, Location, Angle, Scale normalization (BILASnorm)."""
    
    def __init__(self, video_dir, mask_dir, kpt_dir, save_dir=None):
        """
        Args:
            video_dir: Directory containing raw RGB videos.
            mask_dir: Directory containing hand segmentation masks.
            kpt_dir: Directory containing keypoint JSON files.
            save_dir: Directory to save normalized outputs.
        """
        self.video_dir = video_dir
        self.mask_dir = mask_dir
        self.kpt_dir = kpt_dir
        self.save_dir = save_dir
        self.video_list = os.listdir(video_dir)
        self.video_list.sort()

    def get_img(self, video_dir, video, image, transfer=False):
        """
        Load and preprocess an image frame.
        Override this method according to your data format.

        Args:
            video_dir: Root directory of videos.
            video: Video name.
            image: Image file name.
            transfer: If True, replace '.jpg' suffix with '.png'.

        Returns:
            RGB image array resized to (200,200).
        """
        image = image.replace('jpg', 'png') if transfer else image
        img_path = os.path.join(os.path.join(video_dir, video), image)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (200, 200))
        return img

    def get_kpt(self, video, image, max=None):
        """
        Retrieve keypoints for a specific frame.
        Override according to your keypoint storage format.

        Args:
            video: Video name.
            image: Image frame name.
            max: Scaling factor (e.g., original resolution / 200).

        Returns:
            Array of shape (21,2).
        """
        kpt_path = os.path.join(self.kpt_dir, video + '.json')
        with open(kpt_path) as f:
            data = json.load(f)
            kpt_frames = data['info']
        kpt = np.array(kpt_frames[image]['keypoints'])
        if max:
            kpt = kpt * 200 // max
        return kpt

    def get_kpt_stru(self, video):
        """Load the full keypoint JSON structure for a video."""
        kpt_path = os.path.join(self.kpt_dir, video + '.json')
        with open(kpt_path) as f:
            data = json.load(f)
        return data

    def background_norm(self, img, mask, show=False, save=False, video_name=None, image_name=None):
        """Remove background using hand segmentation mask."""
        if np.max(mask) > 1:
            mask[mask <= 200] = 0
            mask[mask > 200] = 1
        else:
            mask[mask > 0.9] = 1
            mask[mask <= 0.9] = 0
        mask = cv2.resize(mask, (200, 200))
        bg_norm = np.zeros_like(img)
        bg_norm[mask > 0] = img[mask > 0]
        if show:
            plt.imshow(bg_norm)
            plt.show()
        if save:
            save_dir = os.path.join(os.path.join(self.save_dir, 'bg_norm'), self.video_name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            imageio.imwrite(os.path.join(save_dir, self.image_name), bg_norm)
        return bg_norm

    def illumination_norm(self, img, mask, target_brightness=120, show=False, save=False, video_name=None, image_name=None):
        """Normalize illumination of the hand region to a target brightness."""
        light_norm = img.copy()
        brightness = np.mean(light_norm[mask > 0])
        coe = target_brightness / brightness
        light_norm[mask > 0] = light_norm[mask > 0] * coe
        max_ = np.max(light_norm[mask > 0])
        light_norm[light_norm > max_] = light_norm[light_norm > max_] * 255 / max_
        light_norm = light_norm.astype(np.uint8)
        if show:
            plt.imshow(light_norm)
            plt.show()
        if save:
            save_dir = os.path.join(os.path.join(self.save_dir, 'light_norm'), self.video_name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            imageio.imwrite(os.path.join(save_dir, self.image_name), light_norm)
        return light_norm

    def location_norm(self, img, kpt, mask=None, show=False, save=False, video_name=None, image_name=None):
        """
        Translate the palm root keypoint to a fixed position (100,190) in a 200x200 canvas.
        """
        root_coord = kpt[0, :]
        bias = root_coord - np.array((100, 190))
        M = np.float32([[1, 0, -bias[0]], [0, 1, -bias[1]]])
        shifted = cv2.warpAffine(img, M, (200, 200))
        if mask is not None:
            shifted_mask = cv2.warpAffine(mask, M, (200, 200)) * 255
        else:
            shifted_mask = None
        kpt_shifted = kpt - bias
        if show:
            vis_pose(kpt_shifted, shifted)
        if save:
            save_dir = os.path.join(os.path.join(self.save_dir, 'loc_norm'), self.video_name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            imageio.imwrite(os.path.join(save_dir, self.image_name), shifted)
        return shifted, kpt_shifted, shifted_mask

    def angle_norm(self, img, kpt, mask=None, show=False, save=False, video_name=None, image_name=None):
        """
        Rotate the hand so that the line from palm root to middle finger base becomes vertical.
        """
        root_coord = kpt[0, :]
        mid_coord = kpt[9, :]
        tan = (mid_coord[0] - root_coord[0]) / (root_coord[1] - mid_coord[1])
        inv = np.degrees(np.arctan(tan))
        M = cv2.getRotationMatrix2D((int(root_coord[0]), int(root_coord[1])), inv, 1)
        rotated = cv2.warpAffine(img, M, (200, 200), borderValue=(0, 0, 0))
        if mask is not None:
            rotated_mask = cv2.warpAffine(mask, M, (200, 200), borderValue=(0, 0, 0))
        else:
            rotated_mask = None
        M = cv2.getRotationMatrix2D((0, 0), -inv, 1)
        kpt = np.matmul(kpt, M[:, :2])
        kpt_rotated = kpt - (kpt[0, :] - root_coord)
        if show:
            vis_pose(kpt_rotated, rotated)
        if save:
            save_dir = os.path.join(os.path.join(self.save_dir, 'angle_norm'), self.video_name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            imageio.imwrite(os.path.join(save_dir, self.image_name), rotated)
        return rotated, kpt_rotated, rotated_mask

    def scale_norm(self, img, kpt, mask=None, show=False, save=False, video_name=None, image_name=None):
        """
        Normalize palm length (root to middle finger base) to a standard size.
        """
        sta = 80.0  # Standard palm length
        root_coord = kpt[0, :]
        mid_coord = kpt[9, :]
        palm_len = euclid_distance(root_coord, mid_coord)
        scale_factor = palm_len / sta
        scaled = cv2.resize(img, (int(img.shape[1] / scale_factor), int(img.shape[0] / scale_factor)))
        if mask is not None:
            scaled_mask = cv2.resize(mask, (int(mask.shape[1] / scale_factor), int(mask.shape[0] / scale_factor)))
        else:
            scaled_mask = None
        kpt = (kpt / scale_factor).astype(np.int32)
        x, y = scaled.shape[0:2]
        if scale_factor > 1:
            scaled_final = np.zeros_like(img, dtype=np.uint8)
            scaled_final[190 - kpt[0, 1]:190 + y - kpt[0, 1], 100 - int(0.5 * x):100 + (x - int(0.5 * x)), :] = scaled
            if scaled_mask is not None:
                mask_final = np.zeros_like(mask, dtype=np.uint8)
                mask_final[190 - kpt[0, 1]:190 + y - kpt[0, 1], 100 - int(0.5 * x):100 + (x - int(0.5 * x)), :] = scaled_mask
            else:
                mask_final = None
            kpt_scaled = kpt + (100 - int(0.5 * x), 190 - kpt[0, 1])
        elif scale_factor < 1:
            scaled_final = scaled[kpt[0, 1] - 190: kpt[0, 1] + 10, int(x * 0.5) - 100: int(x * 0.5) + 100, :]
            if scaled_mask is not None:
                mask_final = scaled_mask[kpt[0, 1] - 190: kpt[0, 1] + 10, int(x * 0.5) - 100: int(x * 0.5) + 100, :]
            else:
                mask_final = None
            kpt_scaled = kpt - (int(0.5 * x) - 100, kpt[0, 1] - 190)
        else:
            scaled_final = scaled
            mask_final = scaled_mask
            kpt_scaled = kpt
        if show:
            vis_pose(kpt_scaled, scaled_final)
        if save:
            save_dir = os.path.join(os.path.join(self.save_dir, 'scale_norm'), self.video_name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            imageio.imwrite(os.path.join(save_dir, self.image_name), scaled_final)
        return scaled_final, kpt_scaled, mask_final

    def BILASnorm(self, img, mask, kpt):
        """Apply all five normalization steps in sequence."""
        bg_norm = self.background_norm(img, mask, save=False, show=False)
        light_norm = self.illumination_norm(bg_norm, mask, save=False, show=False)
        loc_norm, kpt_norm, mask_norm = self.location_norm(light_norm, kpt, mask, save=False, show=False)
        ang_norm, kpt_norm, mask_norm = self.angle_norm(loc_norm, kpt_norm, mask_norm, save=False, show=False)
        sca_norm, kpt_norm, mask_norm = self.scale_norm(ang_norm, kpt_norm, mask_norm, save=False, show=False)
        return sca_norm, kpt_norm

    def process(self, video_list=None, start=0):
        """
        Run BILASnorm on all videos (or a subset).

        Args:
            video_list: List of video names to process. If None, process all.
            start: Starting index or video name string.
        """
        if video_list is None:
            video_list = self.video_list
        if isinstance(start, str):
            start = np.where(np.array(video_list) == start)[0][0]
        for video in video_list[start:]:
            # Skip known problematic samples if needed (uncomment to use)
            # if video in ill_list:
            #     continue
            print(video)
            self.video_name = video
            video_path = os.path.join(self.video_dir, video)
            if not os.path.isdir(video_path):
                continue
            img_list = os.listdir(video_path)
            img_list.sort()
            kpt_dict = self.get_kpt_stru(video)
            for im_name in img_list:
                self.image_name = im_name
                img = self.get_img(self.video_dir, video, im_name)
                mask = self.get_img(self.mask_dir, video, im_name, transfer=False)
                if np.max(mask) < 1:
                    print('mask error')
                    print(im_name)
                    ill_list_real.append(video)
                    continue
                kpt = self.get_kpt(video, im_name, max=480)
                # Apply full normalization pipeline
                sca_norm, kpt_norm = self.BILASnorm(img, mask, kpt)
                save_dir = os.path.join(os.path.join(self.save_dir, 'color_norm2'), self.video_name)
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                imageio.imwrite(os.path.join(save_dir, self.image_name), sca_norm)
                kpt_dict['info'][im_name]['keypoints'] = kpt_norm.tolist()
            kpt_dict['maker'] = 'Real-DHGA'
            kpt_save_path = os.path.join(os.path.join(self.save_dir, 'keypoints_v1_norm'), video + '.json')
            with open(kpt_save_path, 'w') as file:
                json.dump(kpt_dict, file)


if __name__ == '__main__':
    # Example usage: update these paths before running
    video_dir = 'your RGB data directory'
    depth_dir = 'your depth data directory'
    mask_dir = 'your mask data directory'
    kpt_dir = 'your keypoint data directory'
    save_dir = 'saving directory for standardized data'
    bilasNorm = BILASnorm(video_dir, mask_dir, kpt_dir, save_dir=save_dir)
    img_norm = bilasNorm.process()

