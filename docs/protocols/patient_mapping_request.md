# Patient linkage evidence request (unsent)

For the OTFID and FairVLMed custodians separately, request an authorized de-identified mapping and its data dictionary, not direct patient identifiers or medical-record numbers.

Minimum columns: existing image filename, stable cohort-local pseudonymous patient ID, eye/laterality, encounter/visit ID, acquisition time or consistently shifted relative time, and source/site when available. For FairVLMed, also request note encounter ID, note creation/finalization time relative to the prediction index, and a definition of which text was available at that time. Document whether all images/eyes/visits from one patient were kept in one split and how the official split was generated.

Ask whether missing identity fields mean unavailable, withheld, or not applicable; whether multiple images, eyes or visits exist for a patient; and whether IDs remain stable across releases. Retain custodian provenance and usage permission with the mapping privately. Dates must not enable patient re-identification.

A dataset-card count of patients does not substitute for this mapping. Image similarity, shared demographic attributes and matching notes must not be used to fabricate patient IDs. Do not contact custodians or send data until the user authorizes the specific communication.
