# Manuscript working draft

Author: Marchel Andrian Shevchenko. Year: 2026. Software target: v0.7.0.

`main.tex` assembles the introduction, methodology and six verified references using
IEEEtran and IEEE numbered citation style. This is a working protocol, not an accepted
paper or the official CVPR submission template. Switch to the chosen conference's
current author kit before submission. No completed LQNet results are claimed.

Compile in Overleaf with `main.tex` as the main document, or locally:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Required TeX packages: IEEEtran, amsmath, amssymb, bm, cite, booktabs, hyperref.
The LaTeX source is the manuscript deliverable; PDF compilation is reported separately.

## Equation-to-code traceability

| Formulation | Implementation |
|---|---|
| Letterbox, valid rectangle, bilinear RGB / nearest target | `src/ocuweave/data/preprocessing/geometry.py` |
| Encoder and residual spatial mixer | `src/ocuweave/models/vision/compact.py` |
| Multiscale projection, attention, query-mask weighting | `src/ocuweave/models/heads/query_mask.py` |
| Zero-initialized local residual branch | `src/ocuweave/models/local_detail.py` |
| Valid-pixel BCE + image-mean smoothed soft Dice | `src/ocuweave/losses/segmentation.py` |
| Optimization, checkpoint selection, timing and memory | `src/ocuweave/training/engine.py` |
| Original-grid probability restoration | `src/ocuweave/evaluation/original_grid.py` |
| Discrete overlap and empty-mask conventions | `src/ocuweave/evaluation/segmentation.py` |
| Reference roundtrip / component sampling review | `src/ocuweave/data/auditing/review_bundle.py` |

Clinical claims are limited to motivation supported by the cited clinical study.
The dataset record displays a 2021 publication date while the version-3 record was
created in 2024; the bibliography preserves that distinction. Other cited works are
2022–2024. This focused manuscript bibliography does not replace the broader literature review.
