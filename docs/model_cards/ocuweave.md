# OcuWeave model card

## Implemented models

Query and LQNet provide binary OTFID/FIVES segmentation. MultiLabelLQNet provides four independent IDRiD channels (MA, HE, EX, SE). Models initialize randomly; no pretrained clinical weights are distributed. The experimental `OcuWeave` composition additionally accepts byte-token notes and ordered structured features with explicit availability. That composition has synthetic tests only; it is not a pretrained LLM.

## Native backend

A standalone C++17 image-only LQNet implements analytic backward, AdamW and custom CUDA acceleration for convolution/matrix products. The rest of the reference engine runs on the host. Native weights initialize independently of the historical Python backend. Separate full-resolution detail and spatial-query ablations are implemented; they are not ResNet–ViT or mHC variants. Those hybrid designs remain future work.

Native area outputs are pixel counts, bounding boxes, centroids and explicitly denominated image-area percentages. They are not disease probabilities, physical mm², disease severity or validated early-detection outputs. No clinical pretrained weights are released.

## Evidence

Historical OTFID and six supplementary FIVES/IDRiD CPU pilots are recorded in [results](../results/README.md). IDRiD validation Dice/IoU remain zero in the recorded pilots. The software release adds infrastructure and synthetic checks, not improved medical results. Initial training-only T4 diagnostics passed technical checks but failed the small-set fitting goal; see [readiness evidence](../results/gpu-readiness-v0110/README.md). The core experiment matrix remains pending.

## Evaluation and use

New task evaluation uses original dimensions and threshold 0.5, with missing annotations excluded. Empty-reference conventions and denominators are recorded. Provisional groups do not establish patient independence. Pixel entropy in the evidence-conditioned model is uncalibrated; conditioning tokens are not faithful clinical explanations.

Intended use is reproducible research. Patient care, autonomous diagnosis, treatment recommendation, hypertension diagnosis from vessel masks and time-to-vision-loss prediction are outside validated scope. Clinical annotation review, verified patient linkage and independent evaluation remain necessary.

Original code is MIT. Dataset licenses apply independently. Future weight releases require their own provenance, checksums, evaluation and redistribution terms.
