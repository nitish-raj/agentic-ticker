# Security Guide for Agentic Ticker

This document provides comprehensive security guidelines for the Agentic Ticker application, focusing on configuration and secrets management.

## 🔒 Security Overview

Agentic Ticker handles sensitive information including API keys, configuration data, and user information. This guide ensures that all sensitive data is properly secured according to industry best practices.

## ⚠️ IMPORTANT: Configuration Policy

**THIS REPOSITORY USES ENVIRONMENT VARIABLES ONLY**
- **REQUIRED**: Use environment variables for all configuration (especially for Streamlit Cloud)
- **OPTIONAL**: Use .env file for local development (copy from .env.template)
- All API keys must be set as environment variables

## 🚨 Security Features Implemented

### 1. Secure Configuration Management
- Uses environment variables with .env file support for local development (.env.template)
- API keys are stored as environment variables (cloud-compatible)
- No hardcoded secrets in the codebase
- Configuration validation and sanitization

### 2. Input Validation & Sanitization
- Comprehensive input validation framework
- Injection attack prevention (SQL, XSS, command injection)
- Unicode normalization and character validation
- Security middleware for request validation

### 3. API Security
- Secure API key handling
- Rate limiting and request throttling
- Timeout handling for external API calls
- Error message sanitization to prevent information disclosure

### 4. Network Security
- Enhanced CORS configuration
- Security headers enforcement
- Request size limits for DoS protection
- HTTPS/TLS best practices

## 🛡️ Security Features Implemented

### 1. Environment Variable Support
All sensitive configuration can now be set via environment variables:

```bash
# Gemini API Configuration
export GEMINI_API_KEY="your_gemini_api_key_here"
export GEMINI_MODEL="gemini-2.5-flash-lite"
export GEMINI_API_BASE="https://generativelanguage.googleapis.com/v1beta"

# CoinGecko API Configuration  
export COINGECKO_DEMO_API_KEY="your_coingecko_demo_api_key_here"
export COINGECKO_API_KEY="your_coingecko_pro_api_key_here"

# Feature Flags
export ENABLE_WEB_SEARCH="true"
export ENABLE_CRYPTO_ANALYSIS="true"
```

### 2. Configuration Encryption
Optional encryption support for configuration files:
- Uses Fernet symmetric encryption
- Encryption keys managed via environment variables
- Automatic encryption/decryption of sensitive values

### 3. Audit Logging
Comprehensive audit logging for security events:
- Configuration changes
- API key usage
- Security violations
- File permission changes

### 4. Secure File Handling
- Automatic permission setting on configuration files
- Secure backup creation with proper permissions
- Configuration validation and sanitization

## 📋 Security Checklist

### ✅ Before Deployment
- [ ] Set all API keys as environment variables
- [ ] Remove any hardcoded secrets from configuration files
- [ ] Ensure configuration files have secure permissions (600)
- [ ] Run security audit script: `python3 scripts/security_audit.py`
- [ ] Verify config.yaml is properly secured
- [ ] Check git history for exposed secrets

### ✅ Runtime Security
- [ ] API keys are loaded from environment variables
- [ ] Configuration files have restricted permissions
- [ ] Audit logging is enabled and functioning
- [ ] Error messages are sanitized to prevent secret exposure
- [ ] Input validation is enabled

### ✅ Ongoing Maintenance
- [ ] Regularly rotate API keys
- [ ] Monitor audit logs for suspicious activity
- [ ] Update dependencies regularly
- [ ] Run security scans after code changes
- [ ] Review and update security configurations

## 🔧 Configuration Security

### Secure Configuration Template
Use the provided template for secure configuration:

```bash
# Copy the secure template
cp config.yaml.template config.yaml

# Edit config.yaml with your actual values
nano config.yaml
```

### File Permissions
Ensure proper file permissions:

```bash
# Secure configuration file
chmod 600 config.yaml

# Verify permissions
ls -la config.yaml
```

## 🚀 Quick Start for Secure Setup

### 1. Environment Setup
```bash
# Copy template
cp config.yaml.template config.yaml

# Set your API keys in config.yaml
nano config.yaml

# Set secure permissions
chmod 600 config.yaml
```

### 2. Load Environment Variables
```bash
# Load environment variables
# No environment variables needed - using config.yaml only

# Or use python-dotenv in your application
pip install python-dotenv
```

### 3. Run Security Audit
```bash
# Run comprehensive security audit
python3 scripts/security_audit.py

# Should show security score of 80+ and no critical issues
```

## 🔍 Security Audit Tool

The included security audit tool provides comprehensive security analysis:

```bash
python3 scripts/security_audit.py
```

### What it checks:
- ✅ File permissions on configuration files
- ✅ Hardcoded API keys in configuration
- ✅ Git history for exposed secrets
- ✅ Configuration security settings
- ✅ Logging security

### Security Score:
- **80-100**: Good security posture
- **60-79**: Moderate security posture  
- **0-59**: Needs improvement

## 🚨 Incident Response

### If API Keys Are Exposed

1. **Immediate Actions**:
   - Revoke exposed API keys immediately
   - Generate new API keys
   - Update config.yaml
   - Check git history for commits with exposed keys

2. **Git History Cleanup**:
   ```bash
   # Use BFG Repo-Cleaner to remove secrets from git history
   java -jar bfg.jar --replace-text passwords.txt repo.git
   
   # Or use git filter-repo
   git filter-repo --replace-text <(echo "AIzaSy[0-9A-Za-z_-39]===>REDACTED")
   ```

3. **Prevention**:
   - Add pre-commit hooks to prevent secret commits
   - Ensure config.yaml is properly secured
   - Regular security audits

### Security Monitoring

Monitor these files and locations:
- `logs/config_audit.log` - Configuration changes
- Application logs for API key usage
- File permissions on configuration files

## 📚 Best Practices

### 1. Secret Management
- ✅ Use environment variables for all secrets
- ✅ Never commit secrets to version control
- ✅ Rotate API keys regularly
- ✅ Use different keys for different environments
- ✅ Implement principle of least privilege

### 2. Configuration Security
- ✅ Use secure file permissions (600)
- ✅ Validate all configuration inputs
- ✅ Sanitize configuration for logging
- ✅ Backup configurations securely
- ✅ Monitor configuration changes

### 3. Development Security
- ✅ Use different credentials for development
- ✅ Never use production secrets in development
- ✅ Implement security testing in CI/CD
- ✅ Regular dependency updates
- ✅ Code reviews for security issues

## 🔗 Additional Resources

- [OWASP Python Security](https://owasp.org/www-project-cheat-sheets/cheatsheets/Python_Security_Cheat_Sheet.html)
- [Python Secrets Management](https://docs.python.org/3/library/secrets.html)
- [GitGuardian Secret Detection](https://www.gitguardian.com/)
- [Cryptography Best Practices](https://cryptography.io/en/latest/)

## 🆘 Security Support

If you discover a security vulnerability:

1. **Do not** open a public issue
2. Email security details to: security@agentic-ticker.com
3. Include steps to reproduce the vulnerability
4. We'll respond within 48 hours

## 📈 Security Metrics

Track these security metrics:
- Security audit score (target: >80)
- Number of exposed secrets (target: 0)
- Configuration file permissions (target: 600)
- API key rotation frequency (target: quarterly)
- Security incident response time (target: <24 hours)

---

**Remember**: Security is an ongoing process, not a one-time setup. Regular audits and updates are essential for maintaining a secure application.