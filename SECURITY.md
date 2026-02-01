# Security Policy

## Supported Versions

Currently supported versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| main    | Yes                |
| develop | Yes                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow these steps:

### 1. Do NOT Open a Public Issue

Please do not report security vulnerabilities through public GitHub issues.

### 2. Report Privately

Send a detailed report to the repository owner directly or use GitHub Security Advisories.

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### 3. Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity (Critical: 7 days, High: 14 days, Medium: 30 days)

### 4. Disclosure Policy

- We will acknowledge receipt of your report within 48 hours
- We will provide a detailed response within 7 days
- We will notify you when the vulnerability is fixed
- We will publicly disclose the vulnerability after a fix is released (with your permission)

## Security Best Practices

### For Contributors

1. **Never commit secrets**:
   - Use `.env.local` for local secrets (gitignored)
   - Use GitHub Secrets for CI/CD
   - Use environment variables for production

2. **Dependencies**:
   - Keep dependencies up to date
   - Review Dependabot alerts
   - Run `pip-audit` and `npm audit` before commits

3. **Code Security**:
   - Follow OWASP Top 10 guidelines
   - Validate all user inputs
   - Use parameterized queries
   - Implement rate limiting
   - Use HTTPS in production

4. **Authentication**:
   - Use strong password hashing (bcrypt, argon2)
   - Implement session management properly
   - Use secure cookies (httpOnly, secure, sameSite)
   - Implement CSRF protection

### For Deployment

1. **Environment Variables**:
   ```bash
   # Required secrets
   POSTGRES_PASSWORD=<strong-random-password>
   NEXTAUTH_SECRET=<32-char-random-string>
   VAULT_MASTER_KEY=<32-char-random-string>
   ```

2. **Database Security**:
   - Use SSL connections
   - Limit database user permissions
   - Regular backups
   - Enable audit logging

3. **API Security**:
   - Enable CORS with specific origins
   - Implement rate limiting
   - Use API keys for sensitive endpoints
   - Validate and sanitize all inputs

4. **Container Security**:
   - Use non-root users
   - Scan images for vulnerabilities
   - Keep base images updated
   - Minimize image layers

## Security Features

### Implemented

- **Authentication**: NextAuth.js with PostgreSQL backend
- **Password Hashing**: bcrypt with 12 rounds
- **Session Management**: JWT with 30-day expiry
- **Route Protection**: Middleware-based authentication
- **Encrypted Storage**: Fernet encryption for SSH credentials
- **Input Validation**: Pydantic models in FastAPI
- **CORS**: Configured for specific origins
- **SQL Injection Prevention**: Parameterized queries
- **Rate Limiting**: Sliding window algorithm
- **Security Headers**: CSP, HSTS, XFO, etc.
- **Automated Scanning**: CodeQL, Bandit, Trivy, TruffleHog

### In Progress

- [ ] API key authentication
- [ ] Audit logging
- [ ] Two-factor authentication (2FA)
- [ ] WAF integration

## Security Tools

We use the following tools for security:

- **CodeQL**: Static code analysis for Python and JavaScript
- **Dependabot**: Automated dependency updates
- **TruffleHog**: Secret scanning in git history
- **Bandit**: Python security linter
- **Trivy**: Container vulnerability scanning
- **npm audit**: Node.js dependency scanning
- **pip-audit**: Python dependency scanning
- **Safety**: Python dependency vulnerability checker

## Security Contacts

- **Repository Owner**: @Koussaisalem
- **GitHub Security**: Use Security Advisories

## Acknowledgments

We appreciate responsible disclosure. Security researchers who responsibly report vulnerabilities will be acknowledged (with permission).
