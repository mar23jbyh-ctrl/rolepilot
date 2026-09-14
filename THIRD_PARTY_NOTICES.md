# Third-party notices

RolePilot project code is distributed under the [MIT License](LICENSE). This
notice clarifies the treatment of third-party dependencies and test materials;
it does not relicense third-party software or job postings.

- Python and npm dependencies are referenced by `requirements*.txt` and
  `frontend/package-lock.json`. Keep each package's original license and notices
  when distributing the project. `.venv`, `node_modules` and build output are
  not source deliverables in this repository.
- Tesseract and its official language data are external runtime dependencies.
  The installer records the official source and checksum; language binaries are
  Git-ignored and are not claimed as RolePilot-authored assets.
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
