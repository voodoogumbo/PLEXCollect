# PLEXCollect 🎬

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/Powered%20by-OpenAI-black.svg)](https://openai.com/)
[![Plex](https://img.shields.io/badge/Works%20with-Plex-orange.svg)](https://www.plex.tv/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red.svg)](https://streamlit.io/)

**AI-Powered Plex Collection Manager**

PLEXCollect is a local web application that automatically creates and manages Plex collections using AI classification. It scans your Plex library, uses OpenAI to intelligently categorize content, and creates collections like "Halloween Movies", "Christmas Movies", "Marvel Cinematic Universe", and more - all in perfect chronological order for franchises!

⭐ **Star this repo if you find it useful!** ⭐

## ✨ Features

- 🤖 **AI-Powered Classification** - Uses OpenAI o4-mini/GPT models to intelligently categorize your media
- 🎬 **Franchise Chronological Ordering** - Automatically orders franchise movies by story timeline (not release date)
- 📚 **Automatic Collection Management** - Creates and updates Plex collections automatically
- 🌐 **Modern Web Interface** - Beautiful Streamlit web interface for configuration and monitoring
- ⚡ **Ultra-Efficient Mega-Batch Processing** - Process 800+ movies for ~$0.18 (vs $235 with old methods)
- 📊 **Comprehensive Statistics** - Track AI usage, costs, and collection performance
- 🔧 **Flexible Configuration** - Highly configurable categories, prompts, and behavior
- 💾 **Local Storage** - All data stored locally in SQLite database - your data stays private
- 🔄 **Incremental Updates** - Smart scanning to only process new or changed items
- 🎯 **Force Re-classification** - Manual override system for fine-tuning collections

## Prerequisites

- Python 3.8 or higher
- Plex Media Server with authentication token
- OpenAI API key
- Network access to both Plex server and OpenAI API

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/PLEXCollect.git
   cd PLEXCollect
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the application**
   ```bash
   cp config.example.yml config.yaml
   ```
   - Edit `config.yaml` with your details:

   ```yaml
   plex:
     server_url: "http://localhost:32400"  # Your Plex server URL
     token: "your-plex-token-here"         # Your Plex authentication token
     library_sections:                     # Optional: specific sections to scan
       - "Movies"
       - "TV Shows"

   openai:
     api_key: "your-openai-api-key-here"   # Your OpenAI API key
     model: "gpt-4"                        # Model to use (gpt-4, gpt-3.5-turbo, etc.)
     batch_size: 10                        # Items to process per batch
   ```

4. **Get your Plex token**
   - Visit https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/
   - Or use the web interface: Settings → General → Advanced → Copy token

5. **Get your OpenAI API key**
   - Visit https://platform.openai.com/api-keys
   - Create a new API key and add billing information

## Usage

### Starting the Application

```bash
streamlit run main.py
```

The web interface will open in your browser at `http://localhost:8501`

### First Time Setup

1. **Test Connections**
   - Go to Configuration page
   - Test both Plex and OpenAI connections
   - Verify your library sections are detected

2. **Review Categories**
   - Check the Categories page to see default collection categories
   - Categories include: Action Movies, Horror Movies, Christmas Movies, etc.

3. **Run Your First Scan**
   - Go to Library Scan page
   - Select library sections to scan
   - Select categories to process
   - Click "Start Full Scan"

### Web Interface Overview

- **🏠 Dashboard** - Overview of statistics and recent activity
- **⚙️ Configuration** - Test connections and view current settings
- **🔍 Library Scan** - Run scans and monitor progress
- **📊 Categories** - View and manage collection categories
- **📈 Statistics** - AI usage stats, costs, and performance metrics
- **🔧 System** - Database maintenance and system information

## Configuration Options

### Collection Categories

You can customize collection categories in `config.yaml`:

```yaml
collections:
  default_categories:
    - name: "Sci-Fi Movies"
      description: "Science fiction films with futuristic themes"
      prompt: "Is this a science fiction movie with futuristic, space, or advanced technology themes?"
    
    - name: "Feel-Good Movies"
      description: "Uplifting, positive movies that make you feel good"
      prompt: "Is this an uplifting, feel-good movie with positive themes and happy endings?"
```

### OpenAI Settings

- **Model**: Choose between `gpt-4`, `gpt-3.5-turbo`, `gpt-4o`, etc.
- **Batch Size**: Number of items to process per API call (1-20 recommended)
- **Rate Limiting**: Requests per minute and tokens per minute limits
- **Temperature**: AI creativity level (0.0-1.0, lower = more consistent)

### Plex Settings

- **Library Sections**: Specific sections to scan (leave empty for all)
- **Collection Settings**: Auto-create, update behavior, etc.

## 💰 Cost Management & Efficiency

PLEXCollect uses revolutionary **mega-batch optimization** to dramatically reduce AI costs:

### Real-World Results ⚡
- **721 movies** processed across **7 categories** 
- **Total cost: $0.18** (instead of estimated $235 with individual requests)
- **99.9% cost reduction** through smart batching and o4-mini optimization
- **Franchise chronological ordering** automatically applied

### Cost-Saving Features
- **Mega-Batch Processing**: Process 40+ movies per API call
- **Intelligent Chunking**: Automatically splits large libraries
- **Rate Limiting**: Respects OpenAI limits to avoid errors
- **Incremental Scans**: Only processes new/changed items
- **Model Optimization**: Uses efficient o4-mini by default
- **Real-Time Cost Tracking**: Monitor spending in the web interface

### Estimated Costs (with mega-batch optimization)
- **Small library** (< 1,000 items): $0.05-0.25 per full scan
- **Medium library** (1,000-5,000 items): $0.25-1.25 per full scan  
- **Large library** (> 5,000 items): $1.25+ per full scan

*Compare this to $30-235+ with traditional per-movie API calls!*

## Troubleshooting

### Common Issues

1. **Plex Connection Failed**
   - Verify Plex server URL and port
   - Check authentication token
   - Ensure network connectivity

2. **OpenAI API Errors**
   - Verify API key is correct and active
   - Check billing account has funds
   - Monitor rate limits in usage dashboard

3. **Collections Not Created**
   - Verify `auto_create` is enabled in config
   - Check Plex permissions
   - Review scan logs for errors

4. **High API Costs**
   - Reduce batch size
   - Use cheaper models like `gpt-3.5-turbo`
   - Run incremental scans instead of full scans

### Logs and Debugging

- **Log Location**: `data/plexcollect.log` (if file logging enabled)
- **Database**: `data/collections.db` (SQLite database)
- **Debug Mode**: Set logging level to `DEBUG` in config

### Environment Variables

You can override configuration with environment variables:

```bash
export PLEX_SERVER_URL="http://your-plex-server:32400"
export PLEX_TOKEN="your-token"
export OPENAI_API_KEY="your-key"
```

## Advanced Usage

### Custom Categories

Create highly specific categories by customizing the classification prompts:

```yaml
- name: "Heist Movies with Tropical Settings"
  description: "Movies featuring heists in tropical or beach locations"
  prompt: "Is this a heist/crime movie that takes place in a tropical location, beach setting, or island?"
```

### Scheduled Scans

Enable automatic scanning in configuration:

```yaml
scheduling:
  auto_scan_enabled: true
  scan_interval_hours: 24
  scan_time: "02:00"  # 2 AM daily
```

### API Rate Optimization

For large libraries, optimize API usage:

```yaml
openai:
  batch_size: 20          # Maximum items per request
  rate_limit:
    requests_per_minute: 60
    tokens_per_minute: 90000
```

## Development

### Project Structure

```
project_PLEXCollect/
├── main.py                   # Streamlit web application
├── config.yaml              # Configuration file
├── requirements.txt          # Python dependencies
├── api/                      # Core API modules
│   ├── plex_client.py        # Plex server integration
│   ├── openai_client.py      # OpenAI API client
│   ├── collection_manager.py # Main orchestrator
│   └── database.py           # Database operations
├── models/                   # Data models
│   └── database_models.py    # SQLAlchemy models
├── utils/                    # Utility modules
│   ├── config.py             # Configuration management
│   └── logger.py             # Logging utilities
└── data/                     # Local data storage
    ├── collections.db        # SQLite database
    └── plexcollect.log       # Log files
```

### Database Schema

- **MediaItems**: Plex media with metadata
- **CollectionCategories**: Classification rules and prompts
- **ItemClassifications**: AI classification results
- **ScanHistory**: Scan tracking and statistics
- **AIProcessingLog**: API usage and cost tracking

## Security Considerations

- **Local Only**: All data stays on your local machine
- **API Keys**: Store securely, never commit to version control
- **Network**: Consider VPN if accessing remote Plex server
- **Permissions**: Run with minimal required permissions

## 🤝 Contributing

We welcome contributions! PLEXCollect is designed to benefit the entire Plex community.

### Quick Start for Contributors
1. **Fork the repository** on GitHub
2. **Clone your fork** and create a feature branch
3. **Make your changes** and add tests
4. **Test thoroughly** with your own Plex library  
5. **Submit a pull request** with a clear description

### Ways to Contribute
- 🐛 **Bug Reports**: Found an issue? Let us know!
- 💡 **Feature Requests**: Have ideas for new collections or features?
- 🎬 **New Collection Categories**: Share your custom category prompts
- 📚 **Documentation**: Help improve guides and examples
- 🧪 **Testing**: Help test with different library configurations
- 🎨 **UI/UX**: Improve the Streamlit interface

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Popular Contribution Ideas
- Support for additional AI providers (Anthropic Claude, etc.)
- New franchise collections (DC, Horror franchises, etc.)
- Advanced sorting options (by IMDB rating, decade, etc.)
- Integration with other media management tools
- Mobile-responsive UI improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Please respect OpenAI's usage policies and Plex's terms of service when using this software.

## 💬 Support & Community

### Getting Help
1. **📖 Check the Documentation**: Start with this README and troubleshooting section
2. **🔍 Search Issues**: Look through existing GitHub issues for solutions
3. **💬 GitHub Discussions**: Ask questions and share ideas
4. **🐛 Report Bugs**: Create a detailed issue report
5. **📧 Review Logs**: Check `data/plexcollect.log` for error details

### Community Resources
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community chat
- **Contributing Guide**: See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Security Policy**: See [docs/SECURITY.md](docs/SECURITY.md) (coming soon)

### Commercial Support
This is a community-driven open source project. For enterprise features or priority support, consider:
- Contributing to the project
- Sponsoring development
- Hiring contributors for custom implementations

## 🙏 Acknowledgments

### Core Technologies
- [PlexAPI](https://github.com/pkkid/python-plexapi) - Excellent Python library for Plex integration
- [OpenAI API](https://openai.com/api/) - Powerful AI models for intelligent classification
- [Streamlit](https://streamlit.io/) - Amazing framework for building web interfaces
- [SQLAlchemy](https://sqlalchemy.org/) - Robust database operations and ORM

### Inspiration & Community
- The Plex community for endless creativity in media organization
- OpenAI for democratizing access to powerful AI models
- All the contributors who help make this project better

### Special Thanks
- Everyone who helped test and refine the mega-batch optimization
- The franchise movie fans who provided chronological ordering feedback
- Beta testers who helped identify and fix critical bugs

---

## 🌟 Show Your Support

If PLEXCollect has helped organize your media library:

- ⭐ **Star this repository** to help others discover it
- 🐛 **Report bugs** to help improve the project  
- 💡 **Share feature ideas** to guide development
- 🤝 **Contribute code** to make it even better
- 📢 **Tell your friends** about automated Plex collections

**Happy collecting! 🎬✨**

*Made with ❤️ for the Plex community*