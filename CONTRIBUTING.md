# Contributing to Algo

Thank you for considering contributing to this project! We welcome contributions from everyone. Below are the guidelines to help you get started.

## How to Contribute

1. **Fork the Repository**:
   - Click the "Fork" button at the top right of this repository to create your own copy.

2. **Clone Your Fork**:
   - Clone your fork to your local machine using:
     ```bash
     git clone https://github.com/<your-username>/algo.git
     ```

3. **Set Up the Development Environment**:
   - Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - For development dependencies:
     ```bash
     pip install -r requirements-dev.txt
     ```

4. **Create a New Branch**:
   - Create a branch for your feature or bug fix:
     ```bash
     git checkout -b feature/your-feature-name
     ```

5. **Make Changes**:
   - Make your changes and commit them with a clear message:
     ```bash
     git commit -m "Add feature: your feature description"
     ```

6. **Push Your Changes**:
   - Push your branch to your fork:
     ```bash
     git push origin feature/your-feature-name
     ```

7. **Submit a Pull Request**:
   - Go to the original repository and click "New Pull Request."
   - Provide a clear description of your changes.

## Code Style

- Follow [PEP 8](https://pep8.org/) for Python code.
- Use `black` for code formatting:
  ```bash
  black .
  ```
- Sort imports using `isort`:
  ```bash
  isort .
  ```

## Reporting Issues

- Use the [GitHub Issues](https://github.com/Koussaisalem/algo/issues) page to report bugs or request features.
- Provide as much detail as possible, including steps to reproduce the issue.

Thank you for contributing!