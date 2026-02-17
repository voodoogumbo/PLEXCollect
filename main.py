"""PLEXCollect - AI-Powered Plex Collection Manager

Main Streamlit application for managing Plex collections using AI classification.
"""

import asyncio
import logging
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import time
import os

# Configure logging before importing our modules
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class StreamlitLogHandler(logging.Handler):
    """Custom log handler that stores logs in Streamlit session state."""

    def emit(self, record):
        if 'scan_logs' in st.session_state:
            log_entry = {
                'time': datetime.fromtimestamp(record.created).strftime('%H:%M:%S'),
                'level': record.levelname,
                'message': record.getMessage()
            }
            st.session_state.scan_logs.append(log_entry)

            # Keep only last 100 log entries to prevent memory issues
            if len(st.session_state.scan_logs) > 100:
                st.session_state.scan_logs = st.session_state.scan_logs[-100:]

# Import our modules
from utils.config import get_config, config_manager
from api.database import get_database_manager, init_database
from api.plex_client import get_plex_client
from api.openai_client import get_openai_client
from api.collection_manager import get_collection_manager

# Page configuration
st.set_page_config(
    page_title="PLEXCollect",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'scan_running' not in st.session_state:
    st.session_state.scan_running = False
if 'scan_progress' not in st.session_state:
    st.session_state.scan_progress = 0
if 'scan_message' not in st.session_state:
    st.session_state.scan_message = ""
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Dashboard"
if 'scan_logs' not in st.session_state:
    st.session_state.scan_logs = []

def init_app():
    """Initialize the application."""
    try:
        # Setup log handler for UI
        if 'log_handler_added' not in st.session_state:
            log_handler = StreamlitLogHandler()
            log_handler.setLevel(logging.INFO)
            logging.getLogger().addHandler(log_handler)
            st.session_state.log_handler_added = True

        # Initialize database
        init_database()

        # Setup default categories
        try:
            from api.database import get_database_manager
            from utils.config import get_config

            config = get_config()
            db_manager = get_database_manager()

            with db_manager.get_session() as session:
                logging.info("Syncing categories with config file...")

                config_category_names = {cat.name for cat in config.collections.default_categories}

                existing_categories = db_manager.get_collection_categories(session, enabled_only=False)
                existing_category_names = {cat.name for cat in existing_categories}

                # Remove config-sourced categories that are no longer in config
                # (preserve natural language collections)
                for category in existing_categories:
                    if category.name not in config_category_names and getattr(category, 'source', 'config') == 'config':
                        logging.info(f"Removing obsolete category: {category.name}")
                        session.delete(category)

                for category_config in config.collections.default_categories:
                    existing = db_manager.get_category_by_name(session, category_config.name)
                    if existing:
                        existing.description = category_config.description
                        existing.prompt = category_config.prompt
                        logging.info(f"Updated category: {category_config.name}")
                    else:
                        db_manager.create_collection_category(
                            session,
                            name=category_config.name,
                            description=category_config.description,
                            prompt=category_config.prompt,
                            source="config"
                        )
                        logging.info(f"Created new category: {category_config.name}")

                session.commit()
                logging.info("Category sync completed")
        except Exception as category_error:
            logging.warning(f"Could not setup default categories: {category_error}")

        return True
    except Exception as e:
        st.error(f"Failed to initialize application: {e}")
        return False

def test_connections() -> Dict[str, Dict[str, Any]]:
    """Test connections to Plex and AI."""
    results = {}

    # Test Plex connection
    try:
        plex_client = get_plex_client()
        results['plex'] = plex_client.test_connection()
    except Exception as e:
        results['plex'] = {'success': False, 'error': str(e)}

    # Test AI connection
    try:
        openai_client = get_openai_client()
        results['ai'] = openai_client.test_api_connection()
    except Exception as e:
        results['ai'] = {'success': False, 'error': str(e)}

    return results

def render_sidebar():
    """Render the sidebar with navigation and status."""
    st.sidebar.title("🎬 PLEXCollect")
    st.sidebar.markdown("AI-Powered Vibe Collections")

    # Navigation
    st.sidebar.markdown("### Navigation")
    page_options = ["🏠 Dashboard", "⚙️ Configuration", "🔍 Library Scan", "📊 Categories",
                   "🎨 Collection Builder", "📈 Statistics", "🔧 System"]

    current_index = page_options.index(st.session_state.current_page) if st.session_state.current_page in page_options else 0

    selected_page = st.sidebar.selectbox(
        "Choose a page:",
        page_options,
        index=current_index
    )

    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page

    page = st.session_state.current_page

    # Status indicators
    st.sidebar.markdown("### Status")

    try:
        config = get_config()
        st.sidebar.success("✅ Configuration loaded")
    except Exception as e:
        st.sidebar.error("❌ Configuration error")
        st.sidebar.caption(str(e))

    # Connection status (cached for 30 seconds)
    if 'connection_status' not in st.session_state or \
       time.time() - st.session_state.get('last_connection_check', 0) > 30:

        with st.sidebar:
            with st.spinner("Checking connections..."):
                st.session_state.connection_status = test_connections()
                st.session_state.last_connection_check = time.time()

    plex_status = st.session_state.connection_status.get('plex', {})
    if plex_status.get('success'):
        st.sidebar.success("✅ Plex connected")
    else:
        st.sidebar.error("❌ Plex disconnected")

    ai_status = st.session_state.connection_status.get('ai', {})
    if ai_status.get('success'):
        model_name = ai_status.get('model', 'unknown')
        st.sidebar.success(f"✅ AI connected ({model_name})")
    else:
        st.sidebar.error("❌ AI disconnected")

    # Scan status
    if st.session_state.scan_running:
        st.sidebar.warning("🔄 Scan in progress")
        st.sidebar.progress(st.session_state.scan_progress)
        st.sidebar.caption(st.session_state.scan_message)

    return page

def render_dashboard():
    """Render the main dashboard."""
    st.title("🏠 Dashboard")

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)

    try:
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            stats = db_manager.get_database_stats(session)

            with col1:
                st.metric("Media Items", stats.get('media_items', 0))

            with col2:
                st.metric("Categories", stats.get('collection_categories', 0))

            with col3:
                st.metric("Classifications", stats.get('classifications', 0))

            with col4:
                st.metric("Scans", stats.get('scan_history', 0))
    except Exception as e:
        st.error(f"Error loading statistics: {e}")

    # Recent scans
    st.markdown("### Recent Scans")

    try:
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            recent_scans = db_manager.get_recent_scans(session, limit=5)

            if recent_scans:
                scan_data = []
                for scan in recent_scans:
                    scan_data.append({
                        "Started": scan.started_at.strftime("%Y-%m-%d %H:%M"),
                        "Status": scan.status.title(),
                        "Items Scanned": scan.items_scanned,
                        "Collections Updated": scan.collections_created + scan.collections_updated,
                        "Duration": (scan.completed_at - scan.started_at).total_seconds() / 60
                                  if scan.completed_at else "—"
                    })

                df = pd.DataFrame(scan_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No scans found. Run your first scan to get started!")
    except Exception as e:
        st.error(f"Error loading recent scans: {e}")

    # Quick actions
    st.markdown("### Quick Actions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🔍 Quick Scan", use_container_width=True):
            st.session_state.current_page = "🔍 Library Scan"
            st.rerun()

    with col2:
        if st.button("🎨 Collection Builder", use_container_width=True):
            st.session_state.current_page = "🎨 Collection Builder"
            st.rerun()

    with col3:
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.current_page = "⚙️ Configuration"
            st.rerun()

    with col4:
        if st.button("📊 View Categories", use_container_width=True):
            st.session_state.current_page = "📊 Categories"
            st.rerun()

def render_configuration():
    """Render the configuration page."""
    st.title("⚙️ Configuration")

    # Connection testing
    st.markdown("### Connection Test")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Test Plex Connection"):
            with st.spinner("Testing Plex connection..."):
                plex_client = get_plex_client()
                result = plex_client.test_connection()

                if result['success']:
                    st.success("✅ Plex connection successful!")
                    st.json(result)
                else:
                    st.error(f"❌ Plex connection failed: {result.get('error')}")

    with col2:
        if st.button("Test AI Connection"):
            with st.spinner("Testing AI connection..."):
                openai_client = get_openai_client()
                result = openai_client.test_api_connection()

                if result['success']:
                    st.success("✅ AI connection successful!")
                    st.json(result)
                else:
                    st.error(f"❌ AI connection failed: {result.get('error')}")

    # Configuration display
    st.markdown("### Current Configuration")

    try:
        config = get_config()

        with st.expander("Plex Settings"):
            st.write(f"**Server URL:** {config.plex.server_url}")
            st.write(f"**Token:** {'*' * len(config.plex.token) if config.plex.token else 'Not set'}")
            st.write(f"**Library Sections:** {', '.join(config.plex.library_sections) if config.plex.library_sections else 'All'}")

        with st.expander("AI Settings"):
            st.write(f"**Provider:** {config.ai.provider}")
            st.write(f"**Model:** {config.ai.model}")
            st.write(f"**Max Tokens:** {config.ai.max_tokens}")
            st.write(f"**Temperature:** {config.ai.temperature}")
            st.write(f"**Batch Size:** {config.ai.batch_size}")
            st.write(f"**Rate Limit:** {config.ai.rate_limit.requests_per_minute} req/min, {config.ai.rate_limit.tokens_per_minute} tokens/min")

            # Cost estimate
            cost_per_1k = get_openai_client()._get_cost_per_1k_tokens()
            st.write(f"**Estimated Cost:** ${cost_per_1k:.5f} per 1K tokens")

        with st.expander("Collection Settings"):
            st.write(f"**Auto Create:** {config.collections.auto_create}")
            st.write(f"**Update Existing:** {config.collections.update_existing}")
            st.write(f"**Remove Missing:** {config.collections.remove_missing}")

    except Exception as e:
        st.error(f"Error loading configuration: {e}")

    st.markdown("### Edit Configuration")
    st.info("To modify settings, edit the `config.yaml` file and restart the application.")

def render_library_scan():
    """Render the library scan page."""
    st.title("🔍 Library Scan")

    # Scan configuration
    st.markdown("### Scan Configuration")

    col1, col2 = st.columns(2)

    with col1:
        try:
            plex_client = get_plex_client()
            if plex_client.connect():
                sections = plex_client.get_library_sections(['movie', 'show'])
                section_names = [s.title for s in sections]

                selected_sections = st.multiselect(
                    "Library Sections",
                    section_names,
                    default=section_names,
                    help="Select which library sections to scan"
                )
            else:
                st.error("Cannot connect to Plex server")
                selected_sections = []
        except Exception as e:
            st.error(f"Error loading library sections: {e}")
            selected_sections = []

    with col2:
        try:
            db_manager = get_database_manager()
            with db_manager.get_session() as session:
                categories = db_manager.get_collection_categories(session)
                category_names = [c.name for c in categories]

                selected_categories = st.multiselect(
                    "Categories",
                    category_names,
                    default=category_names,
                    help="Select which categories to process"
                )
        except Exception as e:
            st.error(f"Error loading categories: {e}")
            selected_categories = []

    # Advanced options
    st.markdown("### Advanced Options")
    force_reclassify = st.checkbox(
        "🔄 Force re-classification",
        value=False,
        help="Clear existing classifications and re-scan all items."
    )

    if force_reclassify:
        st.warning("⚠️ This will delete all existing classifications for the selected categories and re-classify all items. This may result in additional API costs.")

    # Scan controls
    st.markdown("### Scan Controls")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🚀 Start Full Scan",
                    disabled=st.session_state.scan_running,
                    use_container_width=True):
            if not selected_sections:
                st.error("Please select at least one library section")
            elif not selected_categories:
                st.error("Please select at least one category")
            else:
                start_scan(selected_sections, selected_categories, force_reclassify)

    with col2:
        if st.button("⏹️ Cancel Scan",
                    disabled=not st.session_state.scan_running,
                    use_container_width=True):
            cancel_scan()

    with col3:
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()

    # Scan progress
    if st.session_state.scan_running:
        st.markdown("### Scan Progress")

        progress_bar = st.progress(st.session_state.scan_progress)
        status_text = st.empty()
        status_text.text(st.session_state.scan_message)

        time.sleep(2)
        st.rerun()

    # Log viewer
    st.markdown("### Recent Activity")

    log_container = st.container()
    with log_container:
        if st.session_state.scan_logs:
            recent_logs = st.session_state.scan_logs[-20:]
            log_text = "\n".join([f"{log['time']} - {log['level']} - {log['message']}" for log in recent_logs])
            st.text_area("Scan Logs", log_text, height=200, disabled=True)
        else:
            st.info("No recent activity. Start a scan to see logs here.")

    if st.button("🗑️ Clear Logs"):
        st.session_state.scan_logs = []
        st.rerun()

def start_scan(library_sections: List[str], categories: List[str], force_reclassify: bool = False):
    """Start a library scan."""
    st.session_state.scan_running = True
    st.session_state.scan_progress = 0
    st.session_state.scan_message = "Starting scan..."

    def progress_callback(progress: float, message: str):
        st.session_state.scan_progress = progress
        st.session_state.scan_message = message

    try:
        collection_manager = get_collection_manager()

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(collection_manager.full_scan_and_classify(
            library_sections, categories, force_reclassify, progress_callback
        ))

        st.session_state.scan_running = False
        st.success("Scan completed successfully!")

    except Exception as e:
        st.session_state.scan_running = False
        st.error(f"Scan failed: {e}")

def cancel_scan():
    """Cancel the current scan."""
    try:
        collection_manager = get_collection_manager()
        if collection_manager.cancel_scan():
            st.session_state.scan_running = False
            st.warning("Scan cancelled")
        else:
            st.info("No active scan to cancel")
    except Exception as e:
        st.error(f"Error cancelling scan: {e}")

def render_categories():
    """Render the categories management page."""
    st.title("📊 Categories")

    try:
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            categories = db_manager.get_collection_categories(session, enabled_only=False)

            if categories:
                for category in categories:
                    source = getattr(category, 'source', 'config') or 'config'
                    source_badge = "🤖 AI Created" if source == "natural_language" else "📋 Config"
                    status_label = 'Enabled' if category.enabled else 'Disabled'

                    with st.expander(f"{category.name} ({status_label}) — {source_badge}"):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.write(f"**Description:** {category.description}")
                            st.write(f"**Classification Prompt:** {category.prompt}")
                            st.write(f"**Items Count:** {category.item_count}")

                            # Show original NL query for AI-created collections
                            nl_query = getattr(category, 'natural_language_query', None)
                            if nl_query:
                                st.write(f"**Original Query:** \"{nl_query}\"")

                            if category.last_updated:
                                st.write(f"**Last Updated:** {category.last_updated.strftime('%Y-%m-%d %H:%M')}")

                        with col2:
                            st.write(f"**Status:** {'🟢 Enabled' if category.enabled else '🔴 Disabled'}")
                            st.write(f"**Source:** {source_badge}")
                            st.write(f"**Auto Create:** {'✅' if category.auto_create else '❌'}")
                            st.write(f"**Update Existing:** {'✅' if category.update_existing else '❌'}")

                            # Refresh button for NL categories
                            if source == "natural_language" and nl_query:
                                if st.button("🔄 Refresh", key=f"refresh_{category.id}"):
                                    st.session_state.current_page = "🎨 Collection Builder"
                                    st.session_state['prefill_query'] = nl_query
                                    st.rerun()
            else:
                st.info("No categories configured. Check your configuration file.")

    except Exception as e:
        st.error(f"Error loading categories: {e}")

def render_collection_builder():
    """Render the Collection Builder page — the headline feature."""
    st.title("🎨 Collection Builder")
    st.markdown("Describe the collection you want, and AI will search your library to find matching movies.")

    # Query input
    prefill = st.session_state.pop('prefill_query', '')
    query = st.text_area(
        "Describe the collection you want to create...",
        value=prefill,
        height=100,
        placeholder="Examples:\n• movies about found family\n• films with unreliable narrators\n• cozy movies for a rainy Sunday\n• 90s nostalgia trips\n• visually stunning sci-fi"
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        search_clicked = st.button("🔍 Search My Library", use_container_width=True, disabled=not query.strip())

    # Results
    if search_clicked and query.strip():
        with st.spinner("AI is searching your library..."):
            try:
                collection_manager = get_collection_manager()

                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                result = loop.run_until_complete(
                    collection_manager.natural_language_collection(query.strip())
                )

                st.session_state['nl_result'] = result
                st.session_state['nl_query'] = query.strip()

            except Exception as e:
                st.error(f"Search failed: {e}")

    # Display results from session state
    if 'nl_result' in st.session_state:
        result = st.session_state['nl_result']

        if result.get('error') and not result.get('matching_items'):
            st.warning(result['error'])
        else:
            matching = result.get('matching_items', [])
            st.markdown(f"### Found {len(matching)} matching movies (searched {result.get('total_searched', 0)})")

            if result.get('total_cost'):
                st.caption(f"Search cost: ${result['total_cost']:.4f} | Tokens: {result.get('total_tokens', 0)}")

            if matching:
                # Editable collection name
                suggested_name = result.get('suggested_name', 'Custom Collection')
                collection_name = st.text_input("Collection Name", value=suggested_name)

                # Results table
                for item in matching:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{item['title']}** ({item.get('year', '?')})")
                        if item.get('summary'):
                            st.caption(item['summary'][:150] + "..." if len(item['summary']) > 150 else item['summary'])
                    with col2:
                        genres = item.get('genres', [])
                        if genres:
                            st.caption(", ".join(genres[:3]))

                st.divider()

                # Create collection button
                if st.button("📺 Create Collection in Plex", use_container_width=True):
                    with st.spinner("Creating collection..."):
                        try:
                            collection_manager = get_collection_manager()

                            try:
                                loop = asyncio.get_event_loop()
                            except RuntimeError:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)

                            create_result = loop.run_until_complete(
                                collection_manager.natural_language_collection(
                                    st.session_state.get('nl_query', ''),
                                    auto_create=True
                                )
                            )

                            if create_result.get('collection_created'):
                                st.success(f"✅ Collection '{collection_name}' created in Plex!")
                            elif create_result.get('error'):
                                st.warning(create_result['error'])
                            else:
                                st.info("Collection saved to database.")

                        except Exception as e:
                            st.error(f"Failed to create collection: {e}")
            else:
                st.info("No matching movies found. Try a different description!")

    # History section
    st.divider()
    st.markdown("### Previous Collections")

    try:
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            nl_collections = db_manager.get_natural_language_collections(session)

            if nl_collections:
                for coll in nl_collections:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**{coll.name}**")
                        st.caption(f'"{coll.natural_language_query}"')
                    with col2:
                        st.write(f"{coll.item_count} items")
                    with col3:
                        if coll.created_at:
                            st.caption(coll.created_at.strftime("%Y-%m-%d"))
            else:
                st.caption("No AI-created collections yet. Try the search above!")
    except Exception as e:
        st.caption(f"Could not load history: {e}")

def render_statistics():
    """Render the statistics page."""
    st.title("📈 Statistics")

    try:
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            st.markdown("### AI Usage Statistics")

            thirty_days_ago = datetime.now() - timedelta(days=30)
            ai_stats = db_manager.get_ai_usage_stats(session, thirty_days_ago)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Requests", ai_stats['total_requests'])

            with col2:
                st.metric("Success Rate", f"{ai_stats['success_rate']:.1%}")

            with col3:
                st.metric("Total Tokens", f"{ai_stats['total_tokens']:,}")

            with col4:
                st.metric("Estimated Cost", f"${ai_stats['total_cost']:.2f}")

            st.markdown("### Database Statistics")
            db_stats = db_manager.get_database_stats(session)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Media Items", db_stats['media_items'])
                st.metric("Categories", db_stats['collection_categories'])

            with col2:
                st.metric("Classifications", db_stats['classifications'])
                st.metric("Scan History", db_stats['scan_history'])

            with col3:
                st.metric("AI Logs", db_stats['ai_processing_logs'])
                st.metric("AI Collections", db_stats.get('nl_collections', 0))

    except Exception as e:
        st.error(f"Error loading statistics: {e}")

def render_system():
    """Render the system management page."""
    st.title("🔧 System")

    st.markdown("### Database Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧹 Cleanup Old Data"):
            try:
                db_manager = get_database_manager()
                with db_manager.get_session() as session:
                    result = db_manager.cleanup_old_data(session, days_to_keep=90)
                    st.success(f"Cleanup complete: Removed {result['scans_removed']} old scans and {result['ai_logs_removed']} AI logs")
            except Exception as e:
                st.error(f"Cleanup failed: {e}")

    with col2:
        if st.button("🔄 Reload Configuration"):
            try:
                config_manager.reload_config()
                st.success("Configuration reloaded successfully")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to reload configuration: {e}")

    st.markdown("### System Information")

    try:
        config = get_config()

        info_data = {
            "AI Provider": config.ai.provider,
            "AI Model": config.ai.model,
            "Database Path": config.database.path,
            "Log Level": config.logging.level,
            "Log File": config.logging.file_path if config.logging.file_enabled else "Disabled",
            "Backup Enabled": config.database.backup_enabled,
            "Auto Scan": config.scheduling.auto_scan_enabled
        }

        for key, value in info_data.items():
            st.write(f"**{key}:** {value}")

    except Exception as e:
        st.error(f"Error loading system information: {e}")

def main():
    """Main application entry point."""

    # Initialize app
    if not init_app():
        st.stop()

    # Render sidebar and get selected page
    page = render_sidebar()

    # Render selected page
    if page == "🏠 Dashboard":
        render_dashboard()
    elif page == "⚙️ Configuration":
        render_configuration()
    elif page == "🔍 Library Scan":
        render_library_scan()
    elif page == "📊 Categories":
        render_categories()
    elif page == "🎨 Collection Builder":
        render_collection_builder()
    elif page == "📈 Statistics":
        render_statistics()
    elif page == "🔧 System":
        render_system()

if __name__ == "__main__":
    main()
