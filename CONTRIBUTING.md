# Contributing to rough-contact-etm

Thank you for your interest in contributing to `rough-contact-etm`.

This project is a research-oriented open-source workflow for thermo-electro-mechanical simulation of rough surface contact. Contributions are welcome, especially if they improve reproducibility, documentation, numerical robustness, examples, tests, or physical model clarity.

## Scope of contributions

Useful contributions include:

* Bug reports and reproducibility issues.
* Documentation improvements.
* Small examples and benchmark cases.
* Unit tests for electrical, thermal, and post-processing utilities.
* Numerical robustness improvements.
* New post-processing scripts and visualization tools.
* Validation against analytical, numerical, or experimental references.
* Extensions to the physical model, provided that assumptions and limitations are clearly documented.

For substantial model changes, please open an issue first to discuss the proposed approach.

## Development setup

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/chanblog/rough_contact_etm.git
cd rough_contact_etm
python -m pip install --upgrade pip
pip install -e .
pip install pytest numpy scipy h5py matplotlib
```

Some full simulations require Tamaas and should be run in a local environment where Tamaas is properly installed. The lightweight unit tests are designed to run without Tamaas.

## Running tests

Run the test suite with:

```bash
pytest -q
```

The GitHub Actions workflow currently runs lightweight unit tests without Tamaas on multiple Python versions.

## Code style

Please keep the code readable and research-transparent.

* Use clear variable names.
* Keep comments and output messages in English.
* Document physical assumptions when adding or changing a model.
* Avoid hidden hard-coded paths.
* Avoid committing large generated result files unless they are intentionally included as small benchmark data.
* Prefer small, reviewable pull requests.

## Numerical changes

If a contribution changes numerical behavior, please describe:

* What equation, algorithm, or assumption changed.
* Which example or test was used.
* Whether the change affects convergence, stability, runtime, or output values.
* Whether old results remain reproducible.

For new solvers or model extensions, please add at least one minimal test or example.

## Documentation changes

Documentation improvements are strongly encouraged. Good documentation should explain:

* The physical meaning of the parameters.
* The assumptions behind the model.
* The expected input and output files.
* Known limitations.
* How to reproduce a small example.

## Pull requests

Before submitting a pull request, please check that:

1. The code runs locally.
2. `pytest -q` passes.
3. New files have clear names and are placed in appropriate directories.
4. Generated files, caches, and large results are not accidentally committed.
5. The pull request description explains the motivation and main changes.

## Reporting issues

When reporting a bug, please include:

* Operating system.
* Python version.
* Whether Tamaas is installed.
* The command that failed.
* The full error message or traceback.
* A minimal configuration or script that reproduces the issue, if possible.

## License

By contributing to this repository, you agree that your contributions will be licensed under the same license as the project.
