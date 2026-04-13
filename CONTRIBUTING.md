# Contributing to LC-Bayes R2

Thank you for your interest in contributing to the LC-Bayes R2 framework!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/1958126580/LC-Bayes-R2.git
cd LC-Bayes-R2

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install all dependencies
pip install -e ".[dev,docs]"
```

## Running Tests

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=src --cov-report=html
```

## Code Style

- Use type annotations for all public functions.
- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Docstrings must follow [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html).
- All numerical routines must be vectorized via NumPy — no Python-level loops over large arrays.

## Submitting Changes

1. Fork the repository and create a feature branch (`git checkout -b feature/my-feature`).
2. Write tests for any new functionality.
3. Ensure all tests pass (`python -m pytest tests/ -v`).
4. Submit a pull request with a clear description of the changes.

## Reporting Issues

Please use [GitHub Issues](https://github.com/1958126580/LC-Bayes-R2/issues) and include:
- Python version (`python --version`)
- Operating system
- Minimal reproducible example
- Full error traceback

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
