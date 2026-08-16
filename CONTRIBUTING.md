# Contributing to 3GPP RAG Assistant

Thank you for considering contributing to this project! Here are some guidelines to help you get started.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/3gpp-rag-assistant.git
   cd 3gpp-rag-assistant
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Cloud mode: add your GROQ_API_KEY (free at https://console.groq.com/keys)
   # Local mode: set LLM_PROVIDER=ollama — no API key needed
   ```

## Development Workflow

1. Create a new branch for your feature
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and write tests

3. Run tests and linting
   ```bash
   pytest
   black src/
   flake8 src/
   ```

4. Commit your changes
   ```bash
   git commit -m "Add: description of your changes"
   ```

5. Push to your fork and create a Pull Request

## Code Style

- Follow PEP 8 guidelines
- Use `black` for code formatting
- Add docstrings to all functions and classes
- Keep functions small and focused
- Write descriptive variable names

## Testing

- Write unit tests for new features
- Aim for >80% code coverage
- Test edge cases and error handling

## Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Ensure all tests pass
- Update documentation if needed

## Questions?

Feel free to open an issue for any questions or concerns.
