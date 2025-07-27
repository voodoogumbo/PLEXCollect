# PLEXCollect Deployment Guide 🚀

This guide covers deployment options for PLEXCollect beyond the basic local development setup.

## Table of Contents
- [Local Production Deployment](#local-production-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Automated Deployment](#automated-deployment)
- [Monitoring and Maintenance](#monitoring-and-maintenance)

## Local Production Deployment

### System Requirements

#### Minimum Requirements
- **OS**: Linux, macOS, or Windows 10+
- **Python**: 3.8 or higher
- **RAM**: 512MB available
- **Storage**: 1GB free space
- **Network**: Stable internet connection

#### Recommended Specifications
- **OS**: Linux (Ubuntu 20.04+ or CentOS 8+)
- **Python**: 3.10 or higher
- **RAM**: 2GB available
- **Storage**: 5GB free space (for large libraries)
- **Network**: Ethernet connection (for stability)

### Production Setup

#### 1. Create Dedicated User
```bash
# Create a dedicated user for PLEXCollect
sudo useradd -m -s /bin/bash plexcollect
sudo usermod -aG sudo plexcollect  # If admin access needed
su - plexcollect
```

#### 2. Install in Production Location
```bash
# Clone to a stable location
cd /opt
sudo git clone https://github.com/yourusername/PLEXCollect.git
sudo chown -R plexcollect:plexcollect PLEXCollect
cd PLEXCollect

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 3. Secure Configuration
```bash
# Create secure configuration
cp config.example.yml config.yaml
chmod 600 config.yaml  # Owner read/write only
chown plexcollect:plexcollect config.yaml

# Edit configuration with production values
nano config.yaml
```

#### 4. Set Up Data Directory
```bash
# Create secure data directory
mkdir -p data
chmod 700 data  # Owner access only
chown plexcollect:plexcollect data
```

#### 5. Create Systemd Service (Linux)
```bash
# Create service file
sudo nano /etc/systemd/system/plexcollect.service
```

```ini
[Unit]
Description=PLEXCollect AI-Powered Plex Collection Manager
After=network.target

[Service]
Type=simple
User=plexcollect
Group=plexcollect
WorkingDirectory=/opt/PLEXCollect
Environment=PATH=/opt/PLEXCollect/venv/bin
ExecStart=/opt/PLEXCollect/venv/bin/streamlit run main.py --server.port=8501 --server.address=0.0.0.0
Restart=always
RestartSec=10

# Security settings
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/PLEXCollect/data
ReadWritePaths=/opt/PLEXCollect/logs

[Install]
WantedBy=multi-user.target
```

#### 6. Enable and Start Service
```bash
# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable plexcollect
sudo systemctl start plexcollect

# Check status
sudo systemctl status plexcollect
```

### Windows Service Setup

#### Using NSSM (Non-Sucking Service Manager)

1. **Download NSSM** from [nssm.cc](https://nssm.cc/download)
2. **Install PLEXCollect as service**:
```cmd
# Run as Administrator
nssm install PLEXCollect
# In the GUI:
# - Path: C:\Path\To\PLEXCollect\venv\Scripts\streamlit.exe
# - Arguments: run main.py --server.port=8501
# - Startup directory: C:\Path\To\PLEXCollect
```

## Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.10-slim

# Create app user
RUN useradd --create-home --shell /bin/bash plexcollect

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data && chown -R plexcollect:plexcollect data

# Switch to app user
USER plexcollect

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  plexcollect:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./config.yaml:/app/config.yaml:ro
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    networks:
      - plexcollect-net

  # Optional: Reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - plexcollect
    restart: unless-stopped
    networks:
      - plexcollect-net

networks:
  plexcollect-net:
    driver: bridge
```

### Build and Run
```bash
# Build the image
docker build -t plexcollect .

# Run with docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f plexcollect
```

## Cloud Deployment

### Digital Ocean Droplet

#### 1. Create Droplet
- **Size**: 1GB RAM, 1 vCPU (minimum)
- **OS**: Ubuntu 22.04 LTS
- **Options**: Enable monitoring and backups

#### 2. Initial Setup
```bash
# Connect to droplet
ssh root@your-droplet-ip

# Update system
apt update && apt upgrade -y

# Install required packages
apt install -y python3 python3-pip python3-venv git nginx ufw

# Configure firewall
ufw allow ssh
ufw allow 80
ufw allow 443
ufw enable
```

#### 3. Deploy Application
```bash
# Follow production setup steps above
# Configure nginx reverse proxy
```

### AWS EC2 Deployment

#### 1. Launch EC2 Instance
- **AMI**: Amazon Linux 2 or Ubuntu 22.04
- **Instance Type**: t3.micro (free tier eligible)
- **Security Group**: Allow SSH (22), HTTP (80), HTTPS (443)

#### 2. Setup Process
```bash
# For Amazon Linux 2
sudo yum update -y
sudo yum install -y python3 python3-pip git

# For Ubuntu
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git

# Continue with standard deployment
```

### Google Cloud Platform

#### Using Google Cloud Run
```bash
# Build and push container
gcloud builds submit --tag gcr.io/PROJECT-ID/plexcollect

# Deploy to Cloud Run
gcloud run deploy plexcollect \
  --image gcr.io/PROJECT-ID/plexcollect \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8501
```

## Automated Deployment

### Using GitHub Actions

Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy PLEXCollect

on:
  push:
    branches: [ main ]
  release:
    types: [ published ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python tests/run_tests.py
    
    - name: Deploy to server
      uses: appleboy/ssh-action@v0.1.5
      with:
        host: ${{ secrets.HOST }}
        username: ${{ secrets.USERNAME }}
        key: ${{ secrets.PRIVATE_KEY }}
        script: |
          cd /opt/PLEXCollect
          git pull origin main
          source venv/bin/activate
          pip install -r requirements.txt
          sudo systemctl restart plexcollect
```

### Automated Backups
```bash
#!/bin/bash
# backup-plexcollect.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/backups/plexcollect"
APP_DIR="/opt/PLEXCollect"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
cp $APP_DIR/data/collections.db $BACKUP_DIR/collections_$DATE.db

# Backup configuration (without secrets)
cp $APP_DIR/config.example.yml $BACKUP_DIR/config_example_$DATE.yml

# Backup logs
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz $APP_DIR/data/*.log

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.db" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $DATE"
```

## Monitoring and Maintenance

### Health Monitoring

#### System Health Check Script
```bash
#!/bin/bash
# health-check.sh

SERVICE_NAME="plexcollect"
PORT=8501

# Check if service is running
if ! systemctl is-active --quiet $SERVICE_NAME; then
    echo "ERROR: $SERVICE_NAME service is not running"
    exit 1
fi

# Check if port is listening
if ! nc -z localhost $PORT; then
    echo "ERROR: Port $PORT is not accessible"
    exit 1
fi

# Check web interface
if ! curl -f http://localhost:$PORT/_stcore/health > /dev/null 2>&1; then
    echo "ERROR: Web interface health check failed"
    exit 1
fi

echo "OK: PLEXCollect is healthy"
```

#### Monitoring with Prometheus
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'plexcollect'
    static_configs:
      - targets: ['localhost:8501']
    metrics_path: /metrics
    scrape_interval: 30s
```

### Log Management

#### Log Rotation Configuration
```bash
# /etc/logrotate.d/plexcollect
/opt/PLEXCollect/data/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 plexcollect plexcollect
    postrotate
        systemctl reload plexcollect
    endscript
}
```

### Maintenance Tasks

#### Weekly Maintenance Script
```bash
#!/bin/bash
# maintenance.sh

echo "Starting PLEXCollect maintenance..."

# Update system packages
sudo apt update && sudo apt upgrade -y

# Update PLEXCollect
cd /opt/PLEXCollect
git fetch origin
git pull origin main

# Update Python dependencies
source venv/bin/activate
pip install --upgrade -r requirements.txt

# Clean up old data
find data/ -name "*.tmp" -mtime +7 -delete
find data/ -name "*.bak" -mtime +30 -delete

# Restart service
sudo systemctl restart plexcollect

# Verify health
sleep 10
./health-check.sh

echo "Maintenance completed successfully"
```

## Security Considerations

### Production Security Checklist

- [ ] **Firewall configured** with minimal open ports
- [ ] **SSL/TLS enabled** for web interface
- [ ] **Regular security updates** automated
- [ ] **Log monitoring** configured
- [ ] **Backup strategy** implemented
- [ ] **Access controls** in place
- [ ] **API keys rotated** regularly
- [ ] **File permissions** secured

### SSL/TLS Configuration

#### Let's Encrypt with Certbot
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Troubleshooting

### Common Deployment Issues

#### Service Won't Start
```bash
# Check service status
sudo systemctl status plexcollect

# Check logs
sudo journalctl -u plexcollect -f

# Check file permissions
ls -la /opt/PLEXCollect/config.yaml
```

#### Port Already in Use
```bash
# Find process using port
sudo lsof -i :8501

# Kill process if needed
sudo kill -9 PID
```

#### Database Permissions
```bash
# Fix database permissions
sudo chown plexcollect:plexcollect /opt/PLEXCollect/data/
sudo chmod 700 /opt/PLEXCollect/data/
```

### Performance Optimization

#### Memory Usage
```bash
# Monitor memory usage
htop
# Look for python processes

# Adjust Python garbage collection
export PYTHONOPTIMIZE=1
```

#### Database Optimization
```sql
-- SQLite optimization commands
PRAGMA optimize;
PRAGMA auto_vacuum = INCREMENTAL;
PRAGMA journal_mode = WAL;
```

---

## Next Steps

After successful deployment:

1. **Configure automated backups**
2. **Set up monitoring alerts** 
3. **Test disaster recovery procedures**
4. **Document your specific configuration**
5. **Plan regular maintenance windows**

For additional help, see:
- [README.md](../README.md) - General documentation
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Development guidelines
- [docs/SECURITY.md](SECURITY.md) - Security best practices

---

**Happy deploying! 🚀**