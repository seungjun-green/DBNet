import cv2
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset

class SynthTextDataset(Dataset):
    def __init__(self, mat_file, images_dir, gt_prob_binary, gt_threshold, target_size=(640, 640)):
        self.images_dir = images_dir
        self.gt_prob_binary = gt_prob_binary
        self.gt_threshold = gt_threshold
        self.target_size = target_size

        loaded_mat = loadmat(mat_file)

        imnames = loaded_mat['imnames'][0]
        wordBB = loaded_mat['wordBB'][0]

        self.samples = []
        for i in range(len(imnames)):
            image_name = imnames[i][0] 
            wbb = wordBB[i] 
            self.samples.append((image_name, wbb))

        del loaded_mat # free big dict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, wordBB = self.samples[idx]
        image_path = f"{self.images_dir}/{image_name}"

        # load the image
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) / 255.0

        # convert the wordBB to polygons
        polygons = self.convert_wordBB_to_polygons(wordBB)
        ignore_tags = [False] * len(polygons)

        # resize the image and polygons
        image_resized, polygons_resized = self.resize_image_and_polygons(
            image_rgb, polygons, self.target_size
        )


        data = {
            'image': image_resized, # (H, W, 3)      
            'polygons': polygons_resized,
            'ignore_tags': ignore_tags,
            'filename': image_name
        }

        # geneate G_s, G_s_mask
        data = self.gt_prob_binary.process(data)
        # generate G_d, G_d_mask
        data = self.gt_threshold.process(data)

        # convert to tensors
        image_tensor = torch.from_numpy(data['image']).float().permute(2, 0, 1) # (3, H, W)
        g_s = torch.from_numpy(data['g_s']).float() # (1, H, W)          
        g_s_mask = torch.from_numpy(data['g_s_mask']).float() # (H, W)
        g_d = torch.from_numpy(data['g_d']).float() # (H, W)          
        g_d_mask = torch.from_numpy(data['g_d_mask']).float() # (H, W)

        return {
            'image': image_tensor,
            'g_s': g_s,
            'g_s_mask': g_s_mask,
            'g_d': g_d,
            'g_d_mask': g_d_mask,
            'filename': image_name
        }

    @staticmethod
    def convert_wordBB_to_polygons(wordBB):
        """Convert WordBB into a list of polygons"""
        if wordBB.ndim == 3:
            # (2, 4, num_words)
            num_words = wordBB.shape[2]
            polygons = []
            for i in range(num_words):
                vertices = wordBB[:, :, i].T  # (4,2)
                polygons.append(vertices)
            return polygons
        elif wordBB.ndim == 2:
            return [wordBB.T]  # (4,2)
        else:
            raise ValueError(f"Invalid wordBB shape {wordBB.shape}.")

    @staticmethod
    def resize_image_and_polygons(image, polygons, target_size):
        """Resize image and the polygons accordingly"""
        H_orig, W_orig = image.shape[:2]
        W_new, H_new = target_size 

        # resize image
        image_resized = cv2.resize(image, (W_new, H_new))

        # scale polygons
        scale_x = W_new / float(W_orig)
        scale_y = H_new / float(H_orig)

        polygons_resized = []
        for poly in polygons:
            poly_resized = poly.copy()
            poly_resized[:, 0] *= scale_x
            poly_resized[:, 1] *= scale_y
            polygons_resized.append(poly_resized)

        return image_resized, polygons_resized