## Data Preparation
- CAMERA Dataset: 
  - [Training](http://download.cs.stanford.edu/orion/nocs/camera_train.zip)
  - [Test](http://download.cs.stanford.edu/orion/nocs/camera_val25K.zip)
  - [IKEA_backgrounds](http://download.cs.stanford.edu/orion/nocs/ikea_data.zip)
  - [Composed_depths](http://download.cs.stanford.edu/orion/nocs/camera_composed_depth.zip)

- Real Dataset: 
  - [Training](http://download.cs.stanford.edu/orion/nocs/real_train.zip)
  - [Test](http://download.cs.stanford.edu/orion/nocs/real_test.zip)
- Ground truth pose annotation (for an easier evaluation): 
  - [Val&Real_test](http://download.cs.stanford.edu/orion/nocs/gts.zip)
- [Object Meshes](http://download.cs.stanford.edu/orion/nocs/obj_models.zip)

You can `wget` the files and store them under data/. The data folder general structure is shown:
```bash
.
└── data/
    ├── camera/
    │   ├── train
    │   └── val
    ├── real/
    │   ├── train
    │   └── test
    ├── obj_models/
    │   ├── real_test
    │   ├── real_train
    │   ├── train
    │   └── val
    ├── camera_full_depths/
    │   ├── train
    │   └── val
    ├── gts/
    │   ├── real_test
    │   └── val
    └── ikea_data
```

You can find the Mask RCNN pretrained on MS COCO dataset checkpoints in this [download link](https://drive.google.com/uc?export=download&id=1SeNduFmmuFugT-1SE186YEPahM61JrAH), then store them under models/.

## Installation

Current project is using uv to manage dependencies.

Create a virtual environment:
```bash
uv venv --python 3.10
.venv/scripts/activate      # for Windows
source .venv/bin/activate   # for Linux
```

Install the package:
```bash
uv build
uv pip install -e .
```

## Training and Evaluation
Run the train script:
```bash
python scripts/train.py
# or
uv run scripts/train.py
```

Run the demo script:
```bash
python scripts/demo_eval.py \
    --data real_test \      # or val
    --detect \              # set for detection, not set for evaluation
```


## Modifications to be tested:
- increase `IMAGE_PER_GPU`
- replace `SGD` with `Adam`
- replace `batch normalization` with `group normalization` or `instance normalization`