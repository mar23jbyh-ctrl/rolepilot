# Third-party notices

RolePilot project code is distributed under the [MIT License](LICENSE). This
notice clarifies the treatment of third-party dependencies and test materials;
it does not relicense third-party software or job postings.

- Python and npm dependencies are referenced by `requirements*.txt` and
  `frontend/package-lock.json`. Keep each package's original license and notices
  when distributing the project. `.venv`, `node_modules` and build output are
  not source deliverables in this repository.
- Tesseract is an external OCR engine. Official `tessdata_best` Chinese and
  English models are bundled under `assets/ocr/tessdata` for reproducible use,
  with the original [Apache 2.0 license](assets/ocr/LICENSE) and a pinned
  [source/checksum manifest](assets/ocr/models.json). They are not
  RolePilot-authored assets. The Docker image installs Tesseract and a Chinese
  font from Debian packages; their original notices remain in
  `/usr/share/doc/`. Locally installed extra language data under `data/ocr/`
  remains Git-ignored.
- Job requirements used in committed fixtures are limited necessary paraphrases,
  not copies of entire postings. The committed OCR and interview fixtures are
  synthetic and do not redistribute a job post or a candidate record. Any
  separately collected web sources remain subject to their own licenses and
  terms; this repository does not assert ownership of job postings.
- Candidate resumes, projects and competition experiences in public test
  fixtures are explicitly synthetic. They are not real applicant records.
- Model outputs and evaluation traces are not guarantees of factual correctness
  or authorization to redistribute private data. Follow the selected provider's
  terms before storing or sharing additional traces.

No third-party brand endorsement is implied. Contributors must only contribute
material they are entitled to license.
