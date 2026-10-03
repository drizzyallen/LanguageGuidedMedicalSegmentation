# Shared adapter policy

1. Phase 1 manifest copies preserve the Phase 0 sample IDs, image paths, mask
   paths, and content hashes without modifying the frozen Phase 0 manifests.
2. Split membership is authoritative from the text datasets: QaTa uses
   `Train_ID.xlsx`, `Val_ID.xlsx`, and the test annotation workbook; MosMed
   uses its Train, Val, and Test annotation workbooks.
3. Authority keys are normalized to image basenames. Annotation column B
   supplies the report text.
4. QaTa's `mask_` prefix is removed from workbook keys. MosMed keys already
   equal image basenames.
5. Missing files, missing reports, empty reports, duplicate IDs, conflicting
   reports, duplicate manifest IDs, and unexpected splits are fatal errors.
6. Validation and test augmentation is forbidden.
7. Training-time geometry is disabled unless a method-specific adapter also
   supplies a text transformation that preserves laterality and location.
8. Text embeddings may be cached only after documenting that the official
   method's encoder is deterministic/frozen in that context. Cache keys bind
   method, encoder fingerprint, sample ID, and exact report text.
