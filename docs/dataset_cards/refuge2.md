# REFUGE2 — candidate optic disc/cup segmentation cohort

[Fang et al., arXiv:2202.08994, 2022](https://arxiv.org/abs/2202.08994) describes 2,000 fundus images with glaucoma classification, disc/cup segmentation and fovea annotations across devices and image qualities. [Official challenge](https://refuge.grand-challenge.org/Home2020/). Verify currently accessible files, annotation subsets and participation/data terms; a paper-level total is not an acquired labeled training count. Do not silently substitute the original REFUGE release.

Audit palette values and annotation meaning before conversion. Represent whole disc (including cup) and cup as two channels. Check reference nesting and missing annotations; retain contradictions for review instead of automatically correcting labels. Use patient/eye/device metadata only where supplied and verified. Preserve official evaluation partitions; define internal development selection without accessing hidden labels.

Acquisition and the two-channel pipeline are pending. The [experiment protocol](../protocols/cross_task_experiments.md) proposes structural metrics; no clinical diagnosis or prognosis follows solely from those masks. REFUGE2 images must not receive EMR from unrelated FairVLMed patients.
