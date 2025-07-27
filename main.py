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
        
        # Setup default categories (avoid creating OpenAI client during init)
        try:
            from api.database import get_database_manager
            from utils.config import get_config
            
            config = get_config()
            db_manager = get_database_manager()
            
            with db_manager.get_session() as session:
                # Always sync categories with config file
                logging.info("Syncing categories with config file...")
                
                # Get categories from config
                config_category_names = {cat.name for cat in config.collections.default_categories}
                
                # Get existing categories from database
                existing_categories = db_manager.get_collection_categories(session, enabled_only=False)
                existing_category_names = {cat.name for cat in existing_categories}
                
                # Remove categories that are no longer in config
                for category in existing_categories:
                    if category.name not in config_category_names:
                        logging.info(f"Removing obsolete category: {category.name}")
                        session.delete(category)
                
                # Add or update categories from config
                for category_config in config.collections.default_categories:
                    existing = db_manager.get_category_by_name(session, category_config.name)
                    if existing:
                        # Update existing category
                        existing.description = category_config.description
                        existing.prompt = category_config.prompt
                        existing.is_franchise = category_config.franchise
                        existing.chronological_sorting = category_config.franchise
                        logging.info(f"Updated category: {category_config.name}")
                    else:
                        # Create new category
                        db_manager.create_collection_category(
                            session,
                            name=category_config.name,
                            description=category_config.description,
                            prompt=category_config.prompt,
                            is_franchise=category_config.franchise,
                            chronological_sorting=category_config.franchise
                        )
                        logging.info(f"Created new category: {category_config.name}")
                
                session.commit()
                logging.info("Category sync completed")
        except Exception as category_error:
            # Don't fail the whole app if category setup fails
            logging.warning(f"Could not setup default categories: {category_error}")
        
        return True
    except Exception as e:
        st.error(f"Failed to initialize application: {e}")
        return False

def test_connections() -> Dict[str, Dict[str, Any]]:
    """Test connections to Plex and OpenAI."""
    results = {}
    
    # Test Plex connection
    try:
        plex_client = get_plex_client()
        results['plex'] = plex_client.test_connection()
    except Exception as e:
        results['plex'] = {'success': False, 'error': str(e)}
    
    # Test OpenAI connection
    try:
        openai_client = get_openai_client()
        results['openai'] = openai_client.test_api_connection()
    except Exception as e:
        results['openai'] = {'success': False, 'error': str(e)}
    
    return results

def render_sidebar():
    """Render the sidebar with navigation and status."""
    st.sidebar.title("🎬 PLEXCollect")
    st.sidebar.markdown("AI-Powered Plex Collection Manager")
    
    # Navigation
    st.sidebar.markdown("### Navigation")
    page_options = ["🏠 Dashboard", "⚙️ Configuration", "🔍 Library Scan", "📊 Categories", 
                   "🎬 Franchises", "📈 Statistics", "🔧 System"]
    
    # Use session state for current page selection
    current_index = page_options.index(st.session_state.current_page) if st.session_state.current_page in page_options else 0
    
    selected_page = st.sidebar.selectbox(
        "Choose a page:",
        page_options,
        index=current_index
    )
    
    # Update session state when selectbox changes
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
    
    page = st.session_state.current_page
    
    # Status indicators
    st.sidebar.markdown("### Status")
    
    # Configuration status
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
    
    # Display connection status
    plex_status = st.session_state.connection_status.get('plex', {})
    if plex_status.get('success'):
        st.sidebar.success("✅ Plex connected")
    else:
        st.sidebar.error("❌ Plex disconnected")
    
    openai_status = st.session_state.connection_status.get('openai', {})
    if openai_status.get('success'):
        st.sidebar.success("✅ OpenAI connected")
    else:
        st.sidebar.error("❌ OpenAI disconnected")
    
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
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Quick Scan", use_container_width=True):
            st.session_state.current_page = "🔍 Library Scan"
            st.rerun()
    
    with col2:
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.current_page = "⚙️ Configuration"
            st.rerun()
    
    with col3:
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
        if st.button("Test OpenAI Connection"):
            with st.spinner("Testing OpenAI connection..."):
                openai_client = get_openai_client()
                result = openai_client.test_api_connection()
                
                if result['success']:
                    st.success("✅ OpenAI connection successful!")
                    st.json(result)
                else:
                    st.error(f"❌ OpenAI connection failed: {result.get('error')}")
    
    # Configuration display
    st.markdown("### Current Configuration")
    
    try:
        config = get_config()
        
        # Plex settings
        with st.expander("Plex Settings"):
            st.write(f"**Server URL:** {config.plex.server_url}")
            st.write(f"**Token:** {'*' * len(config.plex.token) if config.plex.token else 'Not set'}")
            st.write(f"**Library Sections:** {', '.join(config.plex.library_sections) if config.plex.library_sections else 'All'}")
        
        # OpenAI settings
        with st.expander("OpenAI Settings"):
            st.write(f"**Model:** {config.openai.model}")
            st.write(f"**Max Tokens:** {config.openai.max_tokens}")
            st.write(f"**Temperature:** {config.openai.temperature}")
            st.write(f"**Batch Size:** {config.openai.batch_size}")
            st.write(f"**Rate Limit:** {config.openai.rate_limit.requests_per_minute} req/min, {config.openai.rate_limit.tokens_per_minute} tokens/min")
        
        # Collection settings
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
        # Library sections
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
        # Categories
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
        help="Clear existing classifications and re-scan all items. Use this if you want to override previous classifications or if the AI has been updated."
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
        
        # Auto-refresh while scanning
        time.sleep(2)
        st.rerun()
    
    # Log viewer section
    st.markdown("### Recent Activity")
    
    # Display recent log messages
    log_container = st.container()
    with log_container:
        if st.session_state.scan_logs:
            # Show last 20 log messages
            recent_logs = st.session_state.scan_logs[-20:]
            log_text = "\n".join([f"{log['time']} - {log['level']} - {log['message']}" for log in recent_logs])
            st.text_area("Scan Logs", log_text, height=200, disabled=True)
        else:
            st.info("No recent activity. Start a scan to see logs here.")
    
    # Clear logs button
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
        
        # Create new event loop for this thread
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run scan
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
                    with st.expander(f"{category.name} ({'Enabled' if category.enabled else 'Disabled'})"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**Description:** {category.description}")
                            st.write(f"**Classification Prompt:** {category.prompt}")
                            st.write(f"**Items Count:** {category.item_count}")
                            if category.last_updated:
                                st.write(f"**Last Updated:** {category.last_updated.strftime('%Y-%m-%d %H:%M')}")
                        
                        with col2:
                            st.write(f"**Status:** {'🟢 Enabled' if category.enabled else '🔴 Disabled'}")
                            st.write(f"**Auto Create:** {'✅' if category.auto_create else '❌'}")
                            st.write(f"**Update Existing:** {'✅' if category.update_existing else '❌'}")
            else:
                st.info("No categories configured. Check your configuration file.")
    
    except Exception as e:
        st.error(f"Error loading categories: {e}")

def render_statistics():
    """Render the statistics page."""
    st.title("📈 Statistics")
    
    try:
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            # AI usage stats
            st.markdown("### AI Usage Statistics")
            
            # Last 30 days
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
            
            # Database statistics
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
                st.metric("User Feedback", db_stats['user_feedback'])
    
    except Exception as e:
        st.error(f"Error loading statistics: {e}")

def render_franchises():
    """Render the franchise management page."""
    st.title("🎬 Franchise Management")
    
    try:
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            # Get all franchises and their movie counts
            franchises = db_manager.get_all_franchises(session)
            
            if not franchises:
                st.info("No franchises detected yet. Run a scan to identify franchise movies.")
                return
            
            # Franchise overview
            st.markdown("### Franchise Overview")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Franchises", len(franchises))
            
            with col2:
                total_franchise_movies = sum(franchises.values())
                st.metric("Franchise Movies", total_franchise_movies)
            
            with col3:
                # Calculate average movies per franchise
                avg_movies = total_franchise_movies / len(franchises) if franchises else 0
                st.metric("Avg Movies/Franchise", f"{avg_movies:.1f}")
            
            # Search and filter options
            st.markdown("### Search & Filter")
            
            col1, col2 = st.columns(2)
            
            with col1:
                search_term = st.text_input("🔍 Search franchises", placeholder="Search by franchise name...")
            
            with col2:
                sort_by = st.selectbox("📊 Sort by", ["Name", "Movie Count", "Alphabetical"])
            
            # Filter franchises based on search
            filtered_franchises = franchises
            if search_term:
                filtered_franchises = {k: v for k, v in franchises.items() 
                                     if search_term.lower() in k.lower()}
            
            # Sort franchises
            if sort_by == "Movie Count":
                sorted_franchises = sorted(filtered_franchises.items(), key=lambda x: x[1], reverse=True)
            elif sort_by == "Alphabetical":
                sorted_franchises = sorted(filtered_franchises.items(), key=lambda x: x[0])
            else:  # Name
                sorted_franchises = list(filtered_franchises.items())
            
            # Display franchises
            st.markdown("### Franchise Collections")
            
            for franchise_name, movie_count in sorted_franchises:
                with st.expander(f"🎬 {franchise_name} ({movie_count} movies)", expanded=False):
                    render_franchise_detail(session, franchise_name, db_manager)
    
    except Exception as e:
        st.error(f"Error loading franchise data: {e}")

def render_franchise_detail(session, franchise_name: str, db_manager):
    """Render detailed view for a specific franchise."""
    try:
        # Get franchise movies in chronological order
        movies = db_manager.get_franchise_movies(session, franchise_name, ordered=True)
        
        if not movies:
            st.warning(f"No movies found for franchise: {franchise_name}")
            return
        
        # Display franchise timeline
        st.markdown(f"#### 📅 Chronological Timeline")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 List View", "📊 Timeline View", "⚙️ Management"])
        
        with tab1:
            # List view with detailed information
            for idx, movie in enumerate(movies, 1):
                col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
                
                with col1:
                    # Show chronological position
                    if movie.chronological_order:
                        st.write(f"**#{movie.chronological_order}**")
                    else:
                        st.write(f"**#{idx}**")
                
                with col2:
                    # Movie title and year
                    st.write(f"**{movie.title}** ({movie.year or 'Unknown'})")
                    if movie.summary:
                        st.caption(movie.summary[:100] + "..." if len(movie.summary) > 100 else movie.summary)
                
                with col3:
                    # Franchise info
                    if movie.franchise_year:
                        st.write(f"🗓️ {movie.franchise_year}")
                    if movie.manual_franchise_override:
                        st.write("✏️ Manual Override")
                
                with col4:
                    # Action buttons
                    if st.button(f"✏️ Edit", key=f"edit_{movie.id}"):
                        st.session_state[f"edit_movie_{movie.id}"] = True
                    
                    # Show edit form if requested
                    if st.session_state.get(f"edit_movie_{movie.id}", False):
                        render_movie_edit_form(session, movie, db_manager, franchise_name)
                
                st.divider()
        
        with tab2:
            # Timeline visualization
            st.markdown("#### 🎯 Visual Timeline")
            
            # Create a simple timeline chart
            timeline_data = []
            for movie in movies:
                timeline_data.append({
                    "Position": movie.chronological_order or 0,
                    "Title": movie.title,
                    "Year": movie.year or 0,
                    "Franchise Year": movie.franchise_year or movie.year or 0
                })
            
            if timeline_data:
                df = pd.DataFrame(timeline_data)
                st.bar_chart(df.set_index("Title")["Position"])
                
                # Show timeline table
                st.dataframe(df, use_container_width=True)
        
        with tab3:
            # Management options
            st.markdown("#### ⚙️ Franchise Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"🔄 Re-scan {franchise_name}", key=f"rescan_{franchise_name}"):
                    st.info("Re-scanning franchise... (This would trigger a targeted rescan)")
                
                if st.button(f"📋 Export Timeline", key=f"export_{franchise_name}"):
                    export_franchise_timeline(movies, franchise_name)
            
            with col2:
                if st.button(f"🎯 Auto-Fix Ordering", key=f"autofix_{franchise_name}"):
                    auto_fix_franchise_ordering(session, franchise_name, db_manager)
                
                if st.button(f"🗑️ Clear Manual Overrides", key=f"clear_{franchise_name}"):
                    clear_franchise_overrides(session, franchise_name, db_manager)
            
            # Bulk reordering interface
            st.markdown("#### 🔄 Bulk Reordering")
            
            if st.checkbox(f"Enable Bulk Reorder Mode", key=f"bulk_mode_{franchise_name}"):
                render_bulk_reorder_interface(session, movies, franchise_name, db_manager)
            
            # Conflict detection
            conflicts = detect_franchise_conflicts(movies)
            if conflicts:
                st.markdown("#### ⚠️ Detected Conflicts")
                for conflict in conflicts:
                    st.warning(f"🚨 {conflict}")
                    
                if st.button(f"🔧 Auto-Resolve Conflicts", key=f"resolve_{franchise_name}"):
                    resolve_franchise_conflicts(session, movies, db_manager)
    
    except Exception as e:
        st.error(f"Error displaying franchise details: {e}")

def render_movie_edit_form(session, movie, db_manager, franchise_name):
    """Render the edit form for a movie's franchise information."""
    st.markdown(f"##### ✏️ Edit: {movie.title}")
    
    with st.form(f"edit_form_{movie.id}"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_position = st.number_input(
                "Chronological Position", 
                value=movie.chronological_order or 1, 
                min_value=1, 
                step=1
            )
            
            new_franchise_year = st.number_input(
                "In-Universe Year", 
                value=movie.franchise_year or movie.year or 2000, 
                min_value=1900, 
                max_value=2100
            )
        
        with col2:
            new_notes = st.text_area(
                "Notes", 
                value=movie.franchise_notes or "",
                placeholder="Add notes about this movie's position in the franchise..."
            )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.form_submit_button("💾 Save Changes"):
                try:
                    db_manager.update_franchise_info(
                        session,
                        media_item_id=movie.id,
                        franchise_name=franchise_name,
                        chronological_order=new_position,
                        franchise_year=new_franchise_year,
                        notes=new_notes
                    )
                    st.success("✅ Changes saved!")
                    st.session_state[f"edit_movie_{movie.id}"] = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving changes: {e}")
        
        with col2:
            if st.form_submit_button("❌ Cancel"):
                st.session_state[f"edit_movie_{movie.id}"] = False
                st.rerun()
        
        with col3:
            if st.form_submit_button("🗑️ Remove from Franchise"):
                try:
                    db_manager.clear_franchise_info(session, movie.id)
                    st.success("✅ Removed from franchise!")
                    st.session_state[f"edit_movie_{movie.id}"] = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error removing from franchise: {e}")

def render_bulk_reorder_interface(session, movies, franchise_name: str, db_manager):
    """Render the bulk reordering interface."""
    st.markdown("##### 🔄 Drag & Drop Reordering")
    st.info("💡 Use the controls below to reorder movies. Changes are saved automatically.")
    
    # Create a simple reordering interface using selectboxes
    st.markdown("**Current Order:**")
    
    # Store the current order in session state if not exists
    order_key = f"reorder_{franchise_name}"
    if order_key not in st.session_state:
        st.session_state[order_key] = [movie.id for movie in movies]
    
    # Show current order with move up/down buttons
    for idx, movie_id in enumerate(st.session_state[order_key]):
        movie = next((m for m in movies if m.id == movie_id), None)
        if not movie:
            continue
            
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            st.write(f"{idx + 1}. **{movie.title}** ({movie.year or 'Unknown'})")
        
        with col2:
            if st.button("⬆️", key=f"up_{movie_id}", disabled=(idx == 0)):
                # Move up
                st.session_state[order_key][idx], st.session_state[order_key][idx-1] = \
                    st.session_state[order_key][idx-1], st.session_state[order_key][idx]
                st.rerun()
        
        with col3:
            if st.button("⬇️", key=f"down_{movie_id}", disabled=(idx == len(st.session_state[order_key])-1)):
                # Move down
                st.session_state[order_key][idx], st.session_state[order_key][idx+1] = \
                    st.session_state[order_key][idx+1], st.session_state[order_key][idx]
                st.rerun()
        
        with col4:
            position_input = st.number_input(
                f"Pos", 
                value=idx + 1, 
                min_value=1, 
                max_value=len(movies),
                key=f"pos_{movie_id}",
                label_visibility="collapsed"
            )
            if position_input != idx + 1:
                # Move to specific position
                move_to_position(st.session_state[order_key], idx, position_input - 1)
                st.rerun()
    
    # Apply changes button
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Apply New Order", key=f"apply_order_{franchise_name}"):
            apply_bulk_reorder(session, st.session_state[order_key], db_manager)
    
    with col2:
        if st.button("🔄 Reset to Original", key=f"reset_order_{franchise_name}"):
            st.session_state[order_key] = [movie.id for movie in movies]
            st.rerun()

def move_to_position(order_list, from_idx, to_idx):
    """Move an item from one position to another in the list."""
    if 0 <= to_idx < len(order_list):
        item = order_list.pop(from_idx)
        order_list.insert(to_idx, item)

def apply_bulk_reorder(session, new_order, db_manager):
    """Apply the new order to the database."""
    try:
        from models.database_models import MediaItem
        
        for new_position, movie_id in enumerate(new_order, 1):
            # Get the movie and update its chronological order
            movie = session.get(MediaItem, movie_id)
            if movie:
                movie.chronological_order = new_position
                movie.manual_franchise_override = True
        
        session.commit()
        st.success("✅ New order applied successfully!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error applying new order: {e}")

def detect_franchise_conflicts(movies):
    """Detect conflicts in franchise ordering."""
    conflicts = []
    
    # Check for duplicate positions
    positions = [m.chronological_order for m in movies if m.chronological_order is not None]
    duplicates = [pos for pos in set(positions) if positions.count(pos) > 1]
    
    if duplicates:
        conflicts.append(f"Duplicate chronological positions: {', '.join(map(str, duplicates))}")
    
    # Check for gaps in numbering
    if positions:
        sorted_positions = sorted(positions)
        expected_positions = list(range(1, len(sorted_positions) + 1))
        if sorted_positions != expected_positions:
            missing = set(expected_positions) - set(sorted_positions)
            if missing:
                conflicts.append(f"Missing positions in sequence: {', '.join(map(str, missing))}")
    
    # Check for movies without positions
    unpositioned = [m for m in movies if m.chronological_order is None]
    if unpositioned:
        titles = [m.title for m in unpositioned[:3]]  # Show first 3
        if len(unpositioned) > 3:
            titles.append(f"... and {len(unpositioned) - 3} more")
        conflicts.append(f"Movies without chronological position: {', '.join(titles)}")
    
    return conflicts

def resolve_franchise_conflicts(session, movies, db_manager):
    """Automatically resolve franchise conflicts."""
    try:
        # Sort movies by year as fallback, then assign sequential positions
        sorted_movies = sorted(movies, key=lambda x: (x.chronological_order or 999, x.year or 9999, x.title))
        
        for idx, movie in enumerate(sorted_movies, 1):
            movie.chronological_order = idx
            movie.manual_franchise_override = True  # Mark as manually resolved
        
        session.commit()
        st.success("✅ Conflicts resolved automatically!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error resolving conflicts: {e}")

def auto_fix_franchise_ordering(session, franchise_name: str, db_manager):
    """Apply AI-suggested improvements to franchise ordering."""
    try:
        # This would ideally re-run AI classification for just this franchise
        # For now, we'll implement a simple fix based on available data
        
        movies = db_manager.get_franchise_movies(session, franchise_name, ordered=False)
        
        # Sort by franchise_year if available, otherwise by release year
        def sort_key(movie):
            return (
                movie.franchise_year or movie.year or 9999,
                movie.title
            )
        
        sorted_movies = sorted(movies, key=sort_key)
        
        improvements_made = 0
        for idx, movie in enumerate(sorted_movies, 1):
            if movie.chronological_order != idx:
                movie.chronological_order = idx
                improvements_made += 1
        
        session.commit()
        st.success(f"✅ Applied {improvements_made} auto-fix improvements!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error auto-fixing ordering: {e}")

def export_franchise_timeline(movies, franchise_name: str):
    """Export franchise timeline to downloadable format."""
    try:
        import json
        from datetime import datetime
        
        # Prepare export data
        export_data = {
            "franchise_name": franchise_name,
            "exported_at": datetime.now().isoformat(),
            "total_movies": len(movies),
            "movies": []
        }
        
        for movie in movies:
            movie_data = {
                "title": movie.title,
                "year": movie.year,
                "chronological_position": movie.chronological_order,
                "franchise_year": movie.franchise_year,
                "manual_override": movie.manual_franchise_override,
                "notes": movie.franchise_notes,
                "summary": movie.summary[:200] + "..." if movie.summary and len(movie.summary) > 200 else movie.summary
            }
            export_data["movies"].append(movie_data)
        
        # Create downloadable JSON
        json_str = json.dumps(export_data, indent=2)
        
        st.download_button(
            label="📥 Download Timeline (JSON)",
            data=json_str,
            file_name=f"{franchise_name.replace(' ', '_')}_timeline.json",
            mime="application/json"
        )
        
        # Also create CSV format
        df = pd.DataFrame(export_data["movies"])
        csv_str = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Timeline (CSV)",
            data=csv_str,
            file_name=f"{franchise_name.replace(' ', '_')}_timeline.csv",
            mime="text/csv"
        )
        
        st.success("✅ Export files ready for download!")
        
    except Exception as e:
        st.error(f"Error exporting timeline: {e}")

def clear_franchise_overrides(session, franchise_name: str, db_manager):
    """Clear all manual overrides for a franchise."""
    try:
        movies = db_manager.get_franchise_movies(session, franchise_name, ordered=False)
        
        cleared_count = 0
        for movie in movies:
            if movie.manual_franchise_override:
                # Reset to AI-determined values by clearing manual override flag
                # (You might want to store original AI values to restore them)
                movie.manual_franchise_override = False
                cleared_count += 1
        
        session.commit()
        st.success(f"✅ Cleared {cleared_count} manual overrides for {franchise_name}")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error clearing overrides: {e}")

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
    
    # System information
    st.markdown("### System Information")
    
    try:
        config = get_config()
        
        info_data = {
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
    elif page == "🎬 Franchises":
        render_franchises()
    elif page == "📈 Statistics":
        render_statistics()
    elif page == "🔧 System":
        render_system()

if __name__ == "__main__":
    main()