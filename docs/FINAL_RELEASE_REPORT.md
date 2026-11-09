# SemanticGuard AI — Final Release Report

**Release preparation date:** 2026-11-09

**Project:** SemanticGuard AI, college mini project

## Release scope

The existing FastAPI/React implementation was preserved. Release preparation covered repository hygiene, clean dependency installation, backend/evaluation tests, frontend production build, API and browser workflow verification, screenshots, and project documentation.

## Verification results

- **Clean Python setup:** Created a new Python 3.12 virtual environment in `/tmp/semanticguard-release-venv` and installed `backend/requirements-dev.txt` successfully.
- **Automated tests:** `32 passed` using the clean environment; five non-failing SWIG deprecation warnings were emitted.
- **Clean frontend setup:** `npm ci` completed from `frontend/package-lock.json`; npm reported zero vulnerabilities. npm emitted a lifecycle-script approval warning for esbuild/fsevents, but installation and build both succeeded.
- **Frontend production build:** `npm run build` succeeded after the clean install.
- **Live API/browser flow:** Synthetic DOCX and PDF uploads succeeded; comparison returned semantic/FAISS, Cross-Encoder and hybrid fields with extracted-text offsets; report history returned the saved result; PDF download returned HTTP 200 with `application/pdf` (1,962 bytes). The actual browser result screen was captured in `screenshots/comparison.png`.
- **API/runtime isolation:** The live check used port 8001 and an SQLite database/upload directory under `/tmp`. The existing process on port 8000 and user upload/database files were not modified.

## Evaluation finding

The bundled Phase 4 synthetic held-out test reported:

| Pipeline | Precision | Recall | F1 | Specificity | False-positive rate |
|---|---:|---:|---:|---:|---:|
| FAISS baseline | 0.750 | 1.000 | 0.857 | 0.667 | 0.333 |
| FAISS + Cross-Encoder | 0.750 | 1.000 | 0.857 | 0.667 | 0.333 |
| Hybrid | 0.667 | 0.667 | 0.667 | 0.667 | 0.333 |

Hybrid scoring did not improve this test result and missed the paraphrase example. The one-candidate-per-pair Phase 4 evaluation cannot measure ranking changes; the Phase 3 candidate-corpus benchmark covers ranking separately. Both benchmark sets are small synthetic examples and do not support real-world plagiarism accuracy claims. The frozen hybrid calibration is not calibrated for real assignments.

## Repository hygiene and artifacts

- `.gitignore` excludes virtual environments, Python/test caches, environment files and common secrets, uploaded documents, local SQLite databases/WAL files, model caches, FAISS index contents, frontend dependencies/build output, and temporary logs. `.env.example`, source files, screenshots, and synthetic evaluation deliverables remain eligible for submission.
- `backend/semanticguard.db` and five user-uploaded PDFs remain on disk and are ignored; their contents were not inspected or changed. The unrelated attendance spreadsheet at the project root also remains untouched and is ignored.
- `backend/app/services/preprocessing.py` and the `split_passages` compatibility wrapper are currently unused by the active comparison route, but were retained as existing implementation/compatibility files. `backend/app/schemas/`, `frontend/src/components/`, `frontend/src/pages/`, and `evaluation/test_documents/` are empty placeholders; no implementation files were removed.
- `.venv`, `node_modules`, `dist`, bytecode, and pytest caches are generated local artifacts and excluded by `.gitignore`; they were retained to avoid destructive cleanup.
- The supplied workspace directory has no `.git` metadata. Therefore tracked/untracked status, commit contents, and a Git-based clean-tree check could not be verified. Initialize/use the intended Git repository before committing or submitting an archive.

## Known limits

- Character ranges point into extracted text, not PDF geometry or DOCX page layout.
- Scanned PDFs/OCR, multilingual detection, LLM explanations, and production authentication/deployment are not included.
- Similarity and reranker scores are review signals, not probabilities or evidence of misconduct.
- Dependency requirement ranges are not fully version-locked on the Python side; clean installation was verified for this machine/runtime on the release date.

## Submission commands

Backend tests (run from `backend/`):

```bash
python -m pytest tests ../evaluation/test_evaluation.py ../evaluation/test_phase4.py -q
```

Frontend build (run from `frontend/`):

```bash
npm ci
npm run build
```

The screenshots referenced by the root README are in `docs/screenshots/`. The architecture, setup, API overview, screenshots, evaluation, and limitations are documented in `README.md` and `evaluation/README.md`.
