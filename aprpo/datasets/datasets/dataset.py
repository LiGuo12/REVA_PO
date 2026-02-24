import cv2
import os
import json
import re
from PIL import Image
from torch.utils.data import Dataset
import torch 
import torchvision.transforms as transforms
import numpy as np

# MIMIC-CXR
class MIMIC_Dataset(Dataset):
    def __init__(self, storage_path=None, ann_file=None, split='train'):
        """
        Initialize MIMIC-CXR dataset from coco.json
        Args:
            text_processor: text processor for report processing
            storage_path: root path to MIMIC-CXR dataset
            split: dataset split ('train', 'val', or 'test')
        """
        self.storage_path = storage_path
        self.split = split
        # Define image root path
        self.image_root = os.path.join(storage_path, 'files')
        
        # Load coco.json
        json_path = ann_file

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Get samples for the specified split
        split_data = data.get(split, {})
        # Create data samples list
        self.samples = []

        for sample in split_data:
            image_path = None
            image_path = sample.get('image_path', {})
            if image_path:
                # Construct full image path
                full_image_path = os.path.join(self.image_root, image_path[0])
                
                # Get report from sample
                report = sample.get('report', '')
                
                self.samples.append({
                    'image_path': full_image_path,
                    'report': report,
                    'id': sample.get('id'),
                    'positive_categories': sample.get('positive_categories'),

                    'categories': sample.get('categories'),
                })
        self.cls_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
        ])

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Get a single sample from the dataset
        Args:
            index: index of the sample to retrieve
        Returns:
            dict containing processed image, text input, and image ID
            image: List[PIL],  # For VLLM
            image_cls: Tensor [B, 3, 224, 224], # For retrieval
            text_input: List[str],
            study_id: List[str],
            subject_id: List[str],
    
        """
        data_sample = self.samples[index]
        image, retrieval_img = self.load_and_process_image_cv2(data_sample['image_path'])
    
        # Clean and process report
        caption = self.clean_reports(data_sample['report'])

        return {
            "image": image,
            "image_cls": retrieval_img, 
            "text_input": caption,
            "id": data_sample['id'],
            "positive_categories": data_sample.get('positive_categories', []),
            "categories": data_sample.get('categories'),
        }
    
    def load_and_process_image_cv2(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not load image at path: {image_path}")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        max_dim = max(w, h)

        square_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        offset_x = (max_dim - w) // 2
        offset_y = (max_dim - h) // 2
        square_img[offset_y:offset_y + h, offset_x:offset_x + w] = img

        vllm_img = cv2.resize(square_img, (336, 336), interpolation=cv2.INTER_AREA)

        vllm_img_pil = Image.fromarray(vllm_img)
        
        pil_img = Image.fromarray(square_img)
        tensor_img = self.cls_transform(pil_img)

        return vllm_img_pil, tensor_img

    def clean_reports(self, report):
        """Clean and process the report text"""
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
            .strip().lower().split('. ')
        # Remove all punctuation
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        # Use periods （ . ） to separate short sentences
        report = ' . '.join(tokens) + ' .'
        return report

    def collater(self, samples):
        """
        Collate a list of samples into a batch.

        Args:
            samples (list): List of samples from __getitem__

        Returns:
            dict: Batched samples with:
                - image: Stacked images [B, 2, C, H, W]
                - text_input: List or tensor of processed reports
                - study_id: List of study IDs
        """
        if len(samples) == 0:
            return {}

        # Create batch dictionary
        batch = {
            key: [sample[key] for sample in samples] for key in samples[0].keys()
        }

        # Stack images if present
        if "image" in batch:
            images = batch["image"]
            if torch.is_tensor(images[0]):
                images = torch.stack(images, dim=0)  # [B, 2, C, H, W]
            batch["image"] = images

        # Stack cls images (already tensor)
        if "image_cls" in batch:
            cls_images = batch["image_cls"]
            batch["image_cls"] = torch.stack(cls_images, dim=0)

        # Handle text inputs - stack if tensors, keep as list otherwise
        if "text_input" in batch:
            if torch.is_tensor(batch["text_input"][0]):
                batch["text_input"] = torch.stack(batch["text_input"])
                
        return batch


# IU-X-ray
class IUXray_Dataset(Dataset):
    def __init__(self, storage_path=None, ann_file=None, split='train'):
        """
        Initialize IU-Xray dataset from coco.json
        Args:
            text_processor: text processor for report processing
            storage_path: root path to MIMIC-CXR dataset
            split: dataset split ('train', 'val', or 'test')
        """
        self.storage_path = storage_path
        self.split = split
        self.ann_file = ann_file

        # Define image root path
        self.image_root = os.path.join(storage_path, 'images')
        
        # Load coco.json
        json_path = self.ann_file

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Get samples for the specified split
        split_data = data.get(split, {})
        
        # Create data samples list
        self.samples = []

        for sample in split_data:
            image_paths = sample.get('image_path', {})
            full_image_paths = [os.path.join(self.image_root, p) for p in image_paths if isinstance(p, str)]

            # Get report from sample
            report = sample.get('report', '')
            self.samples.append({
                'image_paths': full_image_paths,
                'report': report,
                'id': sample.get('id'),
                'positive_categories': sample.get('positive_categories'),
                'categories': sample.get('categories'),
            })
        
        self.cls_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
        ])

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Returns:
            dict with:
              image: List[PIL.Image]                # two 336x336 images(for VLLM)
              image_cls: List[Tensor]         # two [3,224,224](for cls)
              text_input: str
              id: any
        """
        data_sample = self.samples[index]
        img_pil_list = []
        img_tensor_list = []

        for p in data_sample['image_paths']:
            vllm_img_pil, tensor_img = self.load_and_process_image_cv2(p)
            img_pil_list.append(vllm_img_pil)
            img_tensor_list.append(tensor_img)

        if len(img_pil_list) == 0 or len(img_tensor_list) == 0 or len(img_pil_list) !=len(img_tensor_list) :
            raise FileNotFoundError(f"No valid images for sample id={data_sample.get('id')}, paths={data_sample['image_paths']}")
        
        # Clean and process report
        caption = self.clean_reports(data_sample['report'])

        return {
            "image": img_pil_list,                   # List[PIL]
            "image_cls": img_tensor_list,      # List[Tensor]
            "text_input": caption,
            "id": data_sample['id'],
            "positive_categories": data_sample.get('positive_categories', []),
            "categories": data_sample.get('categories'),
        }

    def load_and_process_image_cv2(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at path: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        max_dim = max(w, h)
        square_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        offset_x = (max_dim - w) // 2
        offset_y = (max_dim - h) // 2
        square_img[offset_y:offset_y + h, offset_x:offset_x + w] = img
        vllm_img = cv2.resize(square_img, (336, 336), interpolation=cv2.INTER_AREA)
        vllm_img_pil = Image.fromarray(vllm_img)
        pil_img = Image.fromarray(square_img)
        tensor_img = self.cls_transform(pil_img)

        return vllm_img_pil, tensor_img

    def clean_reports(self, report):
        """Clean and process the report text"""        
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report
    
    def collater(self, samples):
        """
        Collate a list of samples into a batch.

        Returns:
            dict:
              - image: List[List[PIL]]                 # [B][2]
              - image_cls: Tensor               #  [B][2], for each one the shape is [3,224,224]
              - text_input: List[str]
              - id: List
        """
        if len(samples) == 0:
            return {}

        batch = {
            "image": [s["image"] for s in samples],                           # List[List[PIL]]
            "image_cls": [s["image_cls"] for s in samples],                   # List[List[Tensor]]
            "text_input": [s["text_input"] for s in samples],                 # List[str]
            "id": [s["id"] for s in samples],                                 # List
            "positive_categories": [s["positive_categories"] for s in samples],                 # List[str]
            "categories": [s["categories"] for s in samples],
        }

        return batch
