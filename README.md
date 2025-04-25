Current project is using uv to manage dependencies.

Create a virtual environment:
```bash
uv venv --python 3.10
```

For Windows:
```bash
.venv/scripts/activate
```

For Linux:
```bash
source .venv/bin/activate
```

Install the package:
```bash
uv build
uv pip install -e .
```

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


Modifications to be tested:
- increase `IMAGE_PER_GPU`
- replace `SGD` with `Adam`
- replace `batch normalization` with `group normalization` or `instance normalization`