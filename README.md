# PLEXCollect

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/Powered%20by-OpenAI-black.svg)](https://openai.com/)
[![Plex](https://img.shields.io/badge/Works%20with-Plex-orange.svg)](https://www.plex.tv/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red.svg)](https://streamlit.io/)

**AI-Powered Vibe Collections for Plex**

PLEXCollect creates collections that tools like Kometa can't: subjective, taste-based groupings powered by AI. Instead of "movies tagged Action on TMDb", think **"cozy movies perfect for a rainy Sunday"** or **"films with unreliable narrators"**. Describe the collection you want in plain English and PLEXCollect builds it from your library.

## What Makes This Different from Kometa?

Kometa (Plex Meta Manager) is great for factual collections pulled from IMDb/TMDb metadata -- holidays, genres, studios, decades. PLEXCollect focuses on what metadata databases *can't* capture:

- **Vibe-based categories**: "Emotional Gut Punches", "Late Night Weird", "Background Vibes"
- **Natural language collection builder**: Type "movies about found family and belonging" and get a curated collection from your own library
- **Subjective AI classification**: Uses GPT-4o-mini to understand tone, mood, and feel -- not just tags

Use Kometa for factual collections. Use PLEXCollect for everything else.

## Features

- **Natural Language Collection Builder** -- Describe any collection in plain English and search your library
- **AI-Powered Vibe Classification** -- Categorize media by mood, tone, and feel using GPT-4o-mini
- **Mega-Batch Processing** -- Process 800+ movies for ~$0.18 with smart batching
- **Automatic Collection Management** -- Creates and updates Plex collections automatically
- **Modern Web Interface** -- Streamlit-based UI for configuration, scanning, and collection building
- **Incremental Updates** -- Only processes new or changed items
- **Cost Tracking** -- Real-time monitoring of AI usage and spending
- **Local Storage** -- All data in SQLite, your data stays private

## Prerequisites

- Python 3.8 or higher
- Plex Media Server with authentication token
- OpenAI API key
- Network access to both Plex server and OpenAI API

## Installation

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
   Edit `config.yaml` with your details:

   ```yaml
   plex:
     server_url: "http://localhost:32400"
     token: "your-plex-token-here"
     library_sections:
       - "Movies"

   ai:
     api_key: "your-openai-api-key-here"
     model: "gpt-4o-mini"              # Recommended: fast and cheap
     batch_size: 10
   ```

5. **Get your Plex token**
   - Visit https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/

6. **Get your OpenAI API key**
   - Visit https://platform.openai.com/api-keys

## Usage

### Starting the Application

```bash
streamlit run main.py
```

The web interface opens at `http://localhost:8501`.

### Collection Builder

The headline feature. Go to the **Collection Builder** page and type what you want:

- "movies about found family and belonging"
- "films with unreliable narrators"
- "cozy movies for a rainy Sunday"
- "visually stunning cinematography"

PLEXCollect searches your library using AI, shows matching results, suggests a collection name, and creates the collection in Plex with one click. Past queries are saved so you can refresh them as your library grows.

### Vibe Categories

PLEXCollect ships with 10 default vibe-based categories:

| Category | What It Captures |
|----------|-----------------|
| Cozy Comfort Movies | Warm, comforting films for a blanket-and-tea evening |
| Mind-Bending Movies | Films that question reality and keep you thinking |
| Visually Stunning | Cinematography-driven films that are a feast for the eyes |
| Date Night Movies | Romantic, fun, or sophisticated films for two |
| Background Vibes | Easy-watching films that don't demand full attention |
| Emotional Gut Punches | Films that hit hard emotionally |
| Hidden Gems | Under-the-radar films that deserve more attention |
| Adrenaline Rush | Non-stop intensity and action |
| Late Night Weird | Surreal, cult, experimental films for after midnight |
| Kids Can Watch Too | Family-appropriate films that adults genuinely enjoy |

Customize these or add your own in `config.yaml`.

### Web Interface

- **Dashboard** -- Overview of statistics and quick actions
- **Configuration** -- Test connections and view settings
- **Library Scan** -- Run scans and monitor progress
- **Collection Builder** -- Natural language collection creation
- **Categories** -- View and manage vibe categories
- **Statistics** -- AI usage, costs, and performance
- **System** -- Database maintenance and info

## Cost Estimates

PLEXCollect uses **gpt-4o-mini** by default ($0.15 per 1M input tokens), with mega-batch optimization that processes ~40 movies per API call.

| Library Size | Estimated Cost per Full Scan |
|-------------|----------------------------|
| < 1,000 items | $0.05 - $0.25 |
| 1,000 - 5,000 items | $0.25 - $1.25 |
| > 5,000 items | $1.25+ |

Incremental scans (only new items) cost a fraction of a full scan. Real-world result: 721 movies across 7 categories for $0.18.

## Configuration

### AI Settings

```yaml
ai:
  provider: "openai"           # Currently supported: openai
  api_key: "sk-..."
  model: "gpt-4o-mini"         # Recommended for cost/quality balance
  max_tokens: 4000
  temperature: 0.3             # Lower = more consistent
  batch_size: 10
  rate_limit:
    requests_per_minute: 20
    tokens_per_minute: 40000
```

### Custom Categories

```yaml
collections:
  default_categories:
    - name: "Heist Movies"
      description: "Clever heist and con artist films"
      prompt: "Is this a heist, con, or caper movie where the plot revolves around pulling off a scheme?"
```

### Environment Variables

Override config values with environment variables:

```bash
export AI_API_KEY="your-key"        # or OPENAI_API_KEY
export AI_MODEL="gpt-4o-mini"       # or OPENAI_MODEL
export PLEX_SERVER_URL="http://your-plex-server:32400"
export PLEX_TOKEN="your-token"
```

## Development

### Project Structure

```
project_PLEXCollect/
├── main.py                    # Streamlit web application
├── config.yaml                # Configuration (from config.example.yml)
├── requirements.txt           # Python dependencies
├── api/
│   ├── plex_client.py         # Plex server integration
│   ├── openai_client.py       # AI client (mega-batch + NL search)
│   ├── collection_manager.py  # Main orchestrator
│   └── database.py            # Database operations
├── models/
│   └── database_models.py     # SQLAlchemy models
├── utils/
│   ├── config.py              # Configuration management
│   └── logger.py              # Logging utilities
├── tests/                     # Test suite
└── data/
    ├── collections.db         # SQLite database
    └── plexcollect.log        # Log files
```

### Running Tests

```bash
python -m pytest tests/ -v
```

### Database Schema

- **MediaItems** -- Plex media with metadata
- **CollectionCategories** -- Classification rules, prompts, and NL queries
- **ItemClassifications** -- AI classification results
- **ScanHistory** -- Scan tracking and statistics
- **AIProcessingLog** -- API usage and cost tracking

## Troubleshooting

1. **Plex Connection Failed** -- Verify server URL, token, and network connectivity
2. **AI API Errors** -- Check API key, billing status, and rate limits
3. **Collections Not Created** -- Verify `auto_create` is enabled and Plex permissions are correct
4. **Logs**: `data/plexcollect.log` | **Database**: `data/collections.db`

## Backward Compatibility

If upgrading from an older version:
- Config files using `openai:` key are automatically mapped to `ai:`
- Environment variables `OPENAI_API_KEY` and `OPENAI_MODEL` still work
- Existing database is migrated automatically on first run

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request

### Contribution Ideas
- Support for additional AI providers (Anthropic Claude, Google Gemini, local models)
- New vibe category presets
- Advanced sorting options
- Mobile-responsive UI improvements

## License

MIT License -- see [LICENSE](LICENSE) for details.

## Acknowledgments

- [PlexAPI](https://github.com/pkkid/python-plexapi) -- Python library for Plex integration
- [OpenAI API](https://openai.com/api/) -- AI models for classification
- [Streamlit](https://streamlit.io/) -- Web interface framework
- [SQLAlchemy](https://sqlalchemy.org/) -- Database ORM

---

*Made for Plex users who want collections based on vibes, not just metadata.*
