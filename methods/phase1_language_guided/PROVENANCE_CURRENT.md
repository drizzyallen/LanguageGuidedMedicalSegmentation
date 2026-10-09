# Phase 1 provenance — current state (addendum to PROVENANCE.yaml)

`PROVENANCE.yaml` is hashed into every Phase 1 run's `run_config.json`
(`input_sha256`), so it is kept byte-for-byte as it was when the 12 reported
runs were trained. Two of its sections describe the archived 2026-09-16 launch
and are superseded here:

| Field in PROVENANCE.yaml | Recorded value (archived launch) | Current value used by all 12 reported runs |
|---|---|---|
| `data_sources.qata.phase1_manifest.sha256` | `b372f232…` | `e12e88a0b570ad143cd6517f35e8d20de0e65e53cbeb46c2c1d4928e3a614a4a` (byte-identical to Phase 0) |
| `data_sources.mosmed.phase1_manifest.sha256` | `9f86cf11…` | `5f2cbe3a7e1cf91cb5dec1697c1d82c65bd8c46860186dfa67f584c5c6d2973e` (byte-identical to Phase 0) |
| `data_sources.*.phase1_manifest.split_authority` | Annotation workbooks | Frozen Phase 0 manifest; workbooks supply report text only |
| `methods.prolearn` | Pinned for reproduction | Removed from the study by the research advisor (2026-10-03); not run |

Unchanged and still current: the pinned upstream commits (LViT
`ba90775e…`, RecLMIS `d3265c8b…`), requirements hashes, the native-preservation
contract, and the text-workbook hashes (also enforced by `adapter.py`).

Verification: every reported run's `run_config.json` records the manifest
hashes above under `input_sha256`, and `PROVENANCE.yaml`'s own hash
(`fbd28b476d105acdd57aec0730820e656ffbd47835ca815cd3a3e726b595e853`) matches the value in each run config.
