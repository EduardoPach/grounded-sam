# grounded-sam

This repository is a simple display of how to combine Grounding DINO and SAM models

## Usage

If you use `pdm` just `pdm install` otherwise create a new environment and do `pip install git+https://github.com/EduardoPach/grounded-sam.git` or clone the repo and do `pip install -e .`

### Inference and Visualization

```python
from grounded_sam.inference import grounded_segmentation
from grounded_sam.plot import plot_detections_plotly, plot_detections

# This can be a path or an URL or a PIL/numpy image
image = ...
# This should be a list of strings representing the labels
labels = ...

image_array, detections = grounded_segmentation(image, labels, threshold=0.3, polygon_refinement=True)

# This will create a plotly figure alternative use plot_detections to use matplotlib
plot_detections_plotly(image_array, detections)
```

## Limitiations

This is just a naive implementation `Grounding SAM` to showcase how simple it is to combine both models using the Hugging Face `transformers` library. This is not a production-ready code and should be used with caution.