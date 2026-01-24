import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers.models.mask2former import Mask2FormerForUniversalSegmentation
from transformers.models.auto.processing_auto import AutoProcessor

# -----------------------------
# Load model and processor
# -----------------------------
processor = AutoProcessor.from_pretrained(
    "facebook/mask2former-swin-large-mapillary-vistas-semantic",
    use_fast=False
)
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-large-mapillary-vistas-semantic"
)
model.eval()

# -----------------------------
# Load ONE Mapillary image
# -----------------------------
'''ls /mnt/raid/matteo/Mapillary_images/213579.0/
1271137506928143.jpg  191240957319763.jpg   3317074675105583.jpg  614160677580550.jpg   845016557119855.jpg   
1395518271401798.jpg  321541697222657.jpg   344739467977292.jpg   823710512558681.jpg   850873216250119.jpg '''
image_path = "/mnt/raid/matteo/Mapillary_images/213579.0/614160677580550.jpg"
image = Image.open(image_path).convert("RGB")

# -----------------------------
# Run segmentation
# -----------------------------
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

segmentation = processor.post_process_semantic_segmentation(
    outputs,
    target_sizes=[image.size[::-1]]
)[0].cpu().numpy()

# -----------------------------
# Plot raw + segmentation
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# (a) Raw image
axes[0].imshow(image)
axes[0].set_title("(a) Raw Mapillary image", fontsize=11)
axes[0].axis("off")

# (b) Segmentation map
axes[1].imshow(image)
axes[1].imshow(segmentation, cmap="tab20", alpha=0.8)
axes[1].set_title("(b) Semantic segmentation output", fontsize=11)
axes[1].axis("off")

# -----------------------------
# Save for LaTeX
# -----------------------------
fig.canvas.draw()
fig.savefig(
    "figures/mapillary_segmentation_example.pdf",
    bbox_inches="tight"
)
plt.close(fig)
