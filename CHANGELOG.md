# Changelog

All notable changes to PLEXCollect will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Open source preparation and documentation
- MIT License for community use
- Comprehensive CONTRIBUTING.md guidelines
- Enhanced README.md with badges and community focus
- Security documentation and policies

## [1.0.0] - 2025-07-27

### Added 🎉
- **Revolutionary mega-batch optimization** - Process 800+ movies for ~$0.18 instead of $235+
- **Franchise chronological ordering** - Automatic story timeline ordering for Star Wars, MCU, etc.
- **o4-mini model support** - Latest OpenAI reasoning model with larger response windows
- **Minimal index-based classification** - 95% token reduction through ultra-efficient response format
- **Force re-classification option** - Override existing classifications for fine-tuning
- **Complete Streamlit web interface** with franchise management
- **Real-time cost tracking** and AI usage statistics
- **Database migration system** for franchise fields
- **Comprehensive test suite** with 4 test modules
- **Thread-safe singleton patterns** for database and client management

### Collections Supported
- 🎄 **Holiday Collections**: Christmas, Halloween, Thanksgiving
- 🎬 **Major Franchises**: Star Wars, MCU, DC, X-Men, Fast & Furious, Indiana Jones, Mission: Impossible, James Bond
- 📊 **Smart categorization** with AI confidence scoring
- 🔄 **Automatic collection updates** in Plex

### Technical Achievements
- **Ultra-efficient batching**: 40 movies per API call vs traditional single-movie requests
- **Intelligent chunking**: Automatic splitting for large libraries (800+ movies)
- **Token economics optimization**: From ~50-100 tokens per match to ~1 token per match
- **o4-mini integration**: Proper parameter handling for reasoning models
- **Database optimization**: Performance indexes and franchise-specific queries
- **Error handling**: Comprehensive retry logic and graceful degradation

### Performance Metrics
- **721 movies** processed in **~9 minutes**
- **$0.18 total cost** vs estimated $235 with old methods
- **99.9% cost reduction** through mega-batch optimization
- **192 franchise movies** automatically detected and ordered
- **8 collections** created with perfect chronological ordering

### Breaking Changes
- Initial public release - no breaking changes from previous versions

## [0.9.0] - 2025-07-26

### Added
- Phase 2 implementation of minimal index-based format
- Complete franchise management UI with timeline views
- Manual override system with bulk reordering
- Conflict detection and resolution for franchise ordering

### Fixed
- Critical mega-batch index errors
- Database method signature conflicts
- JSON parsing errors from token truncation
- Classification matching issues

## [0.8.0] - 2025-07-25

### Added
- JSON-based mega-batch approach replacing complex text prompts
- Collection-based response format for better token efficiency
- Enhanced error logging with full traceback support
- Automatic chunking for large movie libraries

### Changed
- Migrated from complex text responses to structured JSON
- Improved database initialization to prevent multiple inits
- Enhanced franchise detection with chronological positioning

## [0.7.0] - 2025-07-24

### Added
- Mega-batch optimization for cost reduction
- Database optimization with singleton patterns
- Franchise management system with chronological ordering
- Comprehensive testing framework

### Fixed
- Rate limiting and API timeout issues
- Database migration system for new columns
- Memory optimization for large libraries

## [0.6.0] - 2025-07-23

### Added
- Initial Streamlit web interface
- Basic AI classification system
- Plex integration with PlexAPI
- SQLite database with SQLAlchemy ORM

### Features
- Basic collection creation and management
- OpenAI GPT-4 integration
- Configuration management system
- Logging and error handling

---

## Contributing to the Changelog

When contributing to PLEXCollect, please:

1. **Add entries to the [Unreleased] section** for new changes
2. **Use the format**: `### Added/Changed/Fixed/Removed`
3. **Include relevant details** but keep entries concise
4. **Link to issues/PRs** when applicable: `([#123](https://github.com/user/repo/pull/123))`
5. **Move items from Unreleased to a version section** when releasing

### Categories
- **Added** for new features
- **Changed** for changes in existing functionality  
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** in case of vulnerabilities

### Version Release Process
1. Move unreleased items to new version section
2. Add release date in YYYY-MM-DD format
3. Create git tag: `git tag -a v1.0.0 -m "Release v1.0.0"`
4. Update version in relevant files (setup.py, __init__.py, etc.)
5. Create GitHub release with changelog notes