# Main training file
# Note that pre-trained weights are loaded from models directory
# Config may be altered

# Imports
import os
import torch
import numpy as np
import argparse

from nocs.config import Config
from nocs.model_vit import MaskRCNN
from nocs.dataset import NOCSData

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to save model folders
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class Nocs_train_config(Config):
    # config file for nocs training, derives from base config  
    NAME="NOCS_train"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # background + 6 object categories
    MEAN_PIXEL = np.array([[ 120.66209412, 114.70348358, 105.81269836]])

    # IMAGE_MIN_DIM = 480
    # IMAGE_MAX_DIM = 640

    RPN_ANCHOR_SCALES = (16, 32, 48, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 64

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

    WEIGHT_DECAY = 1e-3
    LEARNING_RATE = 1e-4

    COORD_LOSS_SCALE = 1
    
    COORD_USE_BINS = True
    COORD_NUM_BINS = 32
   
    COORD_SHARE_WEIGHTS = False
    COORD_USE_DELTA = False

    COORD_POOL_SIZE = 14
    COORD_SHAPE = [28, 28]

    USE_MINI_MASK = False

#main training   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',  default='0', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    

    config = Nocs_train_config()

    # Defining camera and real directories
    camera_dir = os.path.join('data', 'camera')
    real_dir = os.path.join('data', 'real')


    #  real classes
    coco_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                  'bus', 'train', 'truck', 'boat', 'traffic light',
                  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                  'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                  'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                  'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                  'kite', 'baseball bat', 'baseball glove', 'skateboard',
                  'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                  'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                  'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                  'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                  'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                  'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                  'teddy bear', 'hair drier', 'toothbrush']
    

    # 0, 40, 46, rand, rand, 64, 42
    synset_names = ['BG', #0
                    'bottle', #1
                    'bowl', #2
                    'camera', #3
                    'can',  #4
                    'laptop',#5
                    'mug'#6
                    ]
    
    class_map = {
        'bottle': 'bottle',
        'bowl':'bowl',
        'cup':'mug',
        'laptop': 'laptop',
    }

    

    coco_cls_ids = []
    for coco_cls in class_map:
        ind = coco_names.index(coco_cls)
        coco_cls_ids.append(ind)
    config.display()

    # trained_path = None if training a new model
    # Default mode assumes 7 classes: BG, Bottle, bowl, camera, can, laptop, mug
    model = MaskRCNN(config=config, model_dir=MODEL_DIR)
    if config.GPU_COUNT > 0 and torch.cuda.is_available():
        device = torch.device('cuda')
        
    else:
        device = torch.device('cpu')

    print("Model to:", device)

    model.to(device)
    
    # Load and prep synthetic train data
    synthtrain = NOCSData(synset_names,'train')
    synthtrain.load_camera_scenes(camera_dir)
    synthtrain.load_real_scenes(real_dir)
    synthtrain.prepare(class_map)

    # Load and prep synthetic validation data
    valset = NOCSData(synset_names,'val')
    valset.load_camera_scenes(camera_dir)
    valset.prepare(class_map)


    # Training - Stage 1
    print("Training network heads")
    model.train_model(synthtrain, valset,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                layers='heads')
    
    # Training - Stage 2
    print("Training network all layers")
    model.train_model(synthtrain, valset,
                learning_rate=config.LEARNING_RATE/4,
                epochs=200,
                layers='all')