# Harvard-FairVLMed

Official author repository: https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP
Dataset: https://huggingface.co/datasets/harvardairobotics/FairVLMed
Accessed and structurally audited 2026-10-31. License: CC BY-NC-ND 4.0;
noncommercial research terms and dataset restrictions remain separate from MIT software.

## Acquired files
- `Dataset/dataset.zip`: 2,668,667,218 bytes;
  SHA-256 `01bbfba4d38dbda8da24e5f73ba4e20b3cbfe91ff9dd4eb03e72ef4196ff7153`.
- `ReadMe/data_summary.csv`: SHA-256
  `d3e32b972c3eae1f3d58c3cc3caec00dc91c83b394c0f127e1032d531e226845`.
- Acquisition metadata records the exact Hugging Face revision and file checksums.

## Structural audit
10,000 CSV rows match 10,000 NPZ filenames, with no unmatched rows or unreferenced
NPZ files. Official split counts are 7,000 training, 1,000 validation and 2,000 test.
The original `note` and generated `gpt4_summary` columns are distinct. No original
notes are empty. No clinical text is reproduced in public logs or this card.

6,255 notes contain the word glaucoma. This is a target-leakage screening flag,
not a semantic adjudication. Nineteen distinct nonempty note texts appear across
multiple official splits; templates may explain some matches. Do not infer patient
identity from repeated notes. Preserve official splits until a reviewed alternative
protocol exists; do not silently delete or relabel records.

## Remaining work
Image decoding/integrity, safe NPZ schema handling, duplicate-patient assessment,
clinical note leakage review and ophthalmic endpoint alignment remain unverified.
The structural audit never unpickles NPZ objects. This is a glaucoma image–text
cohort and must never be joined to unrelated OTFID patients.

Status: original archive and table verified in private Drive; structural audit
complete. Not yet cleaned or approved for multimodal training.
