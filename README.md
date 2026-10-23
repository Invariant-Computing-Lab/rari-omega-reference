# rari-omega-reference

CP03R1 source and reproducibility package. Portfolio codes: G01, G03, G04, G05, G06, G07, G08, G09.

## What is included
Recovered RARI reference methods and original tests. Complete V7 campaign, provider runtime and protected confirmation banks are not packaged as new independent results.

## Run the checks
Install the dependencies specified by the upstream requirements/environment files. This package includes exact original source where stated; use Python 3.11 or later.

```sh
python run_checks.py
```

The wrapper copies the source into a temporary directory, records stdout/stderr and returns nonzero when a declared check fails. Its output is `checks/RESULT.json`. It never submits a paper, contacts a publication service or pushes a repository. Review `RELEASE_SPEC.json` for the test scope.

Direct commands, from `.`:
```sh
python -m unittest discover -s tests -v
```

## Version and provenance
Recovered from the fixed source archives listed in the private delivery registry. Existing data and experiment tables remain historical unless a dated rerun says otherwise. `UPSTREAM_README.md`, when present, is the preserved source documentation and can contain old completion statements. Current status is only in `RELEASE_SPEC.json` and `checks/RESULT.json`.

## Limitations
Software tests are not independent biological, clinical, physical or mathematical validation. Review the original study's failed and untested claims before citing results. Missing components are not fabricated. This version is not automatically the latest code from other concurrent workstreams.

## Publication and licensing
This is a private staging package. Existing licence files retain their original meaning; no new blanket licence or third-party redistribution permission is granted. A human maintainer must approve authorship, rights, secrets and target-account settings before public upload.
## Verified packaging revision — 2026-10-20

Recovered RARI reference methods and original tests. Complete V7 campaign, provider runtime and protected confirmation banks are not packaged as new independent results.

Runtime state: `PASS`. Rights and release approval are separate. See `RELEASE_VERIFICATION.json`. No paper or conference submission is included as a new software result.

## Software-only nested inputs
Historical manuscript files were excluded from nested input ZIPs. Scientific core members are byte-identical to their original hashes. `SOFTWARE_ONLY_INPUTS.json` records the old/new container hashes and exact exclusions; historical input manifests remain historical, not current distribution manifests. No paper or conference submission is released here.
