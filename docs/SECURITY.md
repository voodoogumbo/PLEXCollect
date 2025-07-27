# Security Policy

## Supported Versions

PLEXCollect is actively maintained and security updates are provided for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of PLEXCollect seriously. If you discover a security vulnerability, please follow these steps:

### 🚨 For Critical Security Issues

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please:

1. **Email us directly** at [security-email-placeholder] with details
2. **Use the GitHub Security Advisory feature** to report privately
3. **Include the following information**:
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact assessment
   - Any suggested fixes or mitigations

### 📋 What to Include in Your Report

- **Detailed description** of the vulnerability
- **Steps to reproduce** the security issue
- **Potential impact** (data exposure, privilege escalation, etc.)
- **Affected versions** (if known)
- **Environment details** (OS, Python version, dependencies)
- **Proof of concept** (if applicable)

### ⏱️ Response Timeline

- **Initial Response**: Within 48 hours
- **Assessment Complete**: Within 1 week
- **Fix Available**: Within 2 weeks (critical issues)
- **Public Disclosure**: After fix is released and users have time to update

## Security Best Practices

### 🔐 Configuration Security

#### API Keys and Tokens
- **Never commit** `config.yaml` to version control
- **Use strong, unique API keys** from OpenAI
- **Rotate API keys** regularly (quarterly recommended)
- **Monitor API usage** for unauthorized access
- **Use environment variables** in production deployments

#### Plex Authentication
- **Use dedicated Plex tokens** for PLEXCollect (not your main account token)
- **Limit token permissions** to minimum required (library access only)
- **Monitor Plex server logs** for unusual activity
- **Use HTTPS** for Plex server connections when possible

#### Network Security
- **Run on trusted networks** (avoid public Wi-Fi for configuration)
- **Use VPN** when accessing remote Plex servers
- **Consider firewall rules** to restrict access to PLEXCollect web interface
- **Use secure protocols** (HTTPS/TLS) for external connections

### 🔒 Application Security

#### File Permissions
```bash
# Recommended file permissions
chmod 600 config.yaml           # Read/write for owner only
chmod 755 main.py               # Executable for owner, readable for others
chmod 700 data/                 # Private data directory
```

#### Database Security
- **Backup encryption** for database files
- **Regular database backups** (automated recommended)
- **Monitor database size** and disk usage
- **Secure deletion** of old log files and backups

#### Logging and Monitoring
- **Review logs regularly** for suspicious activity
- **Set log rotation** to prevent disk space issues
- **Monitor resource usage** (CPU, memory, network)
- **Watch for API rate limit warnings**

### 🚨 Security Checklist

Before deploying PLEXCollect:

- [ ] **Configuration secured**: `config.yaml` not in version control
- [ ] **API keys rotated**: Fresh OpenAI API key generated
- [ ] **File permissions set**: Restricted access to configuration and data
- [ ] **Network secured**: Running on trusted network or VPN
- [ ] **Monitoring enabled**: Log review process established
- [ ] **Backups configured**: Regular database and configuration backups
- [ ] **Updates planned**: Security update notification system in place

## Known Security Considerations

### 🔍 API Data Handling

**OpenAI API Communication**:
- Movie titles, summaries, and metadata are sent to OpenAI for classification
- No personal user data or authentication tokens are sent to OpenAI
- API responses are logged locally (consider log retention policies)
- Network traffic to OpenAI is encrypted (HTTPS/TLS)

**Plex API Communication**:
- PLEXCollect requires read access to your Plex library metadata
- No modification of media files or critical Plex settings
- Authentication tokens are stored locally in configuration
- Consider using a dedicated Plex user account for PLEXCollect

### 🔐 Data Storage

**Local Data**:
- All classification results stored in local SQLite database
- Database contains movie metadata but no personal information
- Log files may contain API keys if debug logging is enabled
- Temporary files created during processing (cleaned automatically)

**Data Retention**:
- Classification results stored indefinitely (user configurable)
- Log files rotated based on size/age (configurable)
- API usage statistics maintained for cost tracking
- No data transmitted to third parties except OpenAI for classification

### ⚠️ Potential Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **API Key Exposure** | Unauthorized AI usage, cost implications | Secure configuration storage, regular rotation |
| **Plex Token Compromise** | Unauthorized library access | Dedicated tokens, permission restrictions |
| **Log File Exposure** | Metadata disclosure, debug information | Secure file permissions, log rotation |
| **Database Access** | Collection data exposure | File system permissions, backup encryption |
| **Network Interception** | Metadata in transit | VPN usage, HTTPS enforcement |

## Security Updates

### 📢 Notification Process

Security updates will be communicated through:
- **GitHub Security Advisories** (primary channel)
- **GitHub Releases** with security tags
- **CHANGELOG.md** with security section
- **README.md** security notices for critical issues

### 🔄 Update Process

When security updates are released:

1. **Stop PLEXCollect** gracefully
2. **Backup current configuration** and database
3. **Update to latest version** using git or package manager
4. **Review configuration changes** required
5. **Test functionality** with limited scope
6. **Monitor logs** for any issues
7. **Resume normal operations**

## Third-Party Dependencies

### 📦 Dependency Management

PLEXCollect relies on several third-party packages. Security considerations:

- **Regular updates**: Dependencies updated with security patches
- **Vulnerability scanning**: Automated scanning for known CVEs
- **Minimal dependencies**: Only essential packages included
- **Trusted sources**: Packages from PyPI and official repositories

### 🔍 Key Dependencies
- **OpenAI Python SDK**: Official OpenAI client library
- **PlexAPI**: Community-maintained Plex integration
- **Streamlit**: Web interface framework
- **SQLAlchemy**: Database ORM and connection management

## Incident Response

### 🚨 If You Suspect a Security Breach

1. **Immediately stop** the PLEXCollect service
2. **Disconnect** from network if breach is confirmed
3. **Rotate all API keys** and authentication tokens
4. **Review logs** for suspicious activity
5. **Report the incident** using our security contact
6. **Document the timeline** and impact assessment

### 🔧 Recovery Steps

1. **Assess the scope** of the potential breach
2. **Update to latest version** with security patches
3. **Reconfigure with new credentials** (API keys, tokens)
4. **Monitor systems** for continued suspicious activity
5. **Review and strengthen** security practices

## Contact Information

### 📧 Security Contact

For security-related issues, please contact:
- **Primary**: Create a private GitHub Security Advisory
- **Alternative**: Email [security-email-placeholder]
- **Response Time**: Within 48 hours for critical issues

### 🔗 Resources

- [OpenAI Security Practices](https://openai.com/security/)
- [Plex Security Best Practices](https://support.plex.tv/articles/200220957-network-ports-and-protocols/)
- [Python Security Guidelines](https://python-security.readthedocs.io/)
- [OWASP Python Security](https://owasp.org/www-project-python-security/)

---

**Last Updated**: 2025-07-27
**Version**: 1.0.0

*This security policy is regularly reviewed and updated. Please check back for the latest information.*