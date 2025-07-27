# Contributing to PLEXCollect 🎬

Thank you for your interest in contributing to PLEXCollect! This guide will help you get started with contributing to the project.

## Code of Conduct

PLEXCollect is committed to providing a welcoming and inclusive experience for all contributors. By participating in this project, you agree to:

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Be patient with questions and different skill levels

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- A Plex Media Server (for testing)
- OpenAI API key (for testing AI features)

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/PLEXCollect.git
   cd PLEXCollect
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r tests/requirements.txt
   ```

4. **Set up configuration**
   ```bash
   cp config.example.yml config.yaml
   # Edit config.yaml with your Plex and OpenAI credentials
   ```

5. **Run tests to ensure everything works**
   ```bash
   python tests/run_tests.py
   ```

## Types of Contributions

We welcome several types of contributions:

### 🐛 Bug Reports
- Use the GitHub issue tracker
- Include clear steps to reproduce
- Provide system information (Python version, OS, etc.)
- Include relevant log files or error messages

### 💡 Feature Requests
- Check existing issues first to avoid duplicates
- Describe the problem you're trying to solve
- Explain how the feature would benefit users
- Consider proposing an implementation approach

### 🔧 Code Contributions
- Bug fixes
- New features
- Performance improvements
- Documentation improvements
- Test coverage improvements

### 📚 Documentation
- README improvements
- Code comments and docstrings
- Configuration examples
- Troubleshooting guides

## Development Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Your Changes
- Write clean, readable code
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes
```bash
# Run all tests
python tests/run_tests.py

# Run specific test files
python -m pytest tests/test_collection_manager.py -v

# Run the application to test manually
streamlit run main.py
```

### 4. Commit Your Changes
```bash
git add .
git commit -m "feat: add support for custom collection ordering"
```

#### Commit Message Guidelines
Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for adding tests
- `refactor:` for code refactoring
- `style:` for formatting changes
- `perf:` for performance improvements

### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear title and description
- Reference any related issues
- Include screenshots if UI changes are involved
- List breaking changes if any

## Code Style Guidelines

### Python Code Style
- Follow PEP 8 conventions
- Use type hints where appropriate
- Write descriptive variable and function names
- Keep functions focused and concise
- Add docstrings for public functions and classes

### Example Code Style
```python
async def classify_media_items(
    self, 
    media_items: List[Dict[str, Any]], 
    categories: List[Dict[str, Any]],
    progress_callback: Optional[Callable] = None
) -> List[BatchResult]:
    """Classify media items using AI with mega-batch optimization.
    
    Args:
        media_items: List of media item dictionaries
        categories: List of category dictionaries
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of BatchResult objects containing classification results
    """
    # Implementation here
```

### File Organization
- Keep modules focused on a single responsibility
- Use clear file and directory names
- Group related functionality together
- Avoid circular imports

## Testing Guidelines

### Writing Tests
- Write tests for all new functionality
- Use descriptive test names
- Test both success and error cases
- Mock external dependencies (Plex, OpenAI)

### Test Structure
```python
def test_classify_media_items_success():
    """Test successful classification of media items."""
    # Arrange
    media_items = [{"id": 1, "title": "Test Movie"}]
    categories = [{"id": 1, "name": "Test Category"}]
    
    # Act
    results = classifier.classify_media_items(media_items, categories)
    
    # Assert
    assert len(results) == 1
    assert results[0].success is True
```

### Running Tests
```bash
# Run all tests
python tests/run_tests.py

# Run with coverage
python -m pytest --cov=api --cov=utils tests/

# Run specific test
python -m pytest tests/test_openai_client.py::test_classify_batch -v
```

## Documentation Guidelines

### Code Documentation
- Add docstrings to all public functions and classes
- Use clear, concise descriptions
- Include parameter and return types
- Add usage examples for complex functions

### README Updates
- Keep installation instructions current
- Update feature lists when adding new functionality
- Add troubleshooting entries for common issues
- Include configuration examples

## Security Considerations

### Handling Secrets
- Never commit API keys, tokens, or passwords
- Use environment variables or config files (in .gitignore)
- Sanitize logs to remove sensitive information
- Review code for potential secret exposure

### API Safety
- Implement proper rate limiting
- Validate user inputs
- Handle API errors gracefully
- Don't log full API responses (may contain sensitive data)

## Performance Guidelines

### Optimization Principles
- Profile before optimizing
- Focus on algorithmic improvements first
- Use async/await for I/O operations
- Batch API calls when possible
- Cache expensive operations

### AI API Best Practices
- Use appropriate batch sizes
- Implement exponential backoff for retries
- Monitor token usage and costs
- Use efficient prompt designs

## Specific Areas for Contribution

### High-Priority Areas
1. **Additional AI Models**: Support for other AI providers (Anthropic, Cohere, etc.)
2. **Enhanced Franchise Detection**: Better chronological ordering algorithms
3. **Performance Optimization**: Faster scanning and classification
4. **UI Improvements**: Better user experience and visualizations
5. **Testing**: Improved test coverage and integration tests

### Collection Categories
We're always looking for new collection categories! Consider contributing:
- Genre-specific collections
- Decade-based collections
- Director or actor collections
- Award-winning movie collections
- International cinema collections

### AI Prompt Engineering
Help improve classification accuracy by:
- Refining existing prompts
- Adding new classification rules
- Improving franchise detection logic
- Optimizing token usage

## Release Process

### Versioning
We use [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped appropriately
- [ ] Security review completed

## Getting Help

### Questions and Support
- **GitHub Discussions**: For general questions and ideas
- **GitHub Issues**: For bug reports and feature requests
- **Code Review**: We provide helpful feedback on pull requests

### Resources
- [Plex API Documentation](https://python-plexapi.readthedocs.io/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)

## Recognition

We value all contributions! Contributors will be:
- Listed in the project README
- Mentioned in release notes for significant contributions
- Invited to help with project direction and roadmap

## Questions?

Feel free to open an issue or start a discussion if you have any questions about contributing. We're here to help and appreciate your interest in making PLEXCollect better!

---

**Thank you for contributing to PLEXCollect! 🎬✨**