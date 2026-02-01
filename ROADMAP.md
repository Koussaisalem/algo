# Roadmap

## Current Status (v1.0.0)

### Completed Features
- [x] Full-stack web platform (Next.js + FastAPI)
- [x] User authentication with PostgreSQL
- [x] Molecule generation with QCMD-ECS diffusion
- [x] Property-guided generation (band gap targeting)
- [x] Molecule library with search/export
- [x] Cloud training with SSH terminal
- [x] System monitoring and resource detection
- [x] Security scanning (CodeQL, Bandit, TruffleHog)
- [x] Automated dependency updates (Dependabot)
- [x] Rate limiting and security headers
- [x] Input validation and sanitization

---

## Planned Improvements

### Phase 1: Testing & Quality (Priority: High)
**Timeline: Next 2 weeks**

- [ ] **Frontend Tests**
  - Unit tests for components (Jest + React Testing Library)
  - Integration tests for key user flows
  - E2E tests with Playwright/Cypress
  - Target: 70% code coverage

- [ ] **Backend Tests**
  - Expand core tests beyond manifold operations
  - API endpoint integration tests
  - Load testing with Locust
  - Target: 80% code coverage

- [ ] **Code Quality**
  - Pre-commit hooks (husky + lint-staged)
  - Prettier for consistent formatting
  - ESLint strict mode
  - TypeScript strict mode
  - Python type hints with mypy

### Phase 2: API & Documentation (Priority: High)
**Timeline: Next 3 weeks**

- [ ] **API Documentation**
  - Interactive Swagger/OpenAPI docs (already in FastAPI)
  - API client SDK generation
  - Postman collection
  - Rate limit documentation

- [ ] **Developer Documentation**
  - Architecture decision records (ADR)
  - API integration guide
  - Deployment guide (production-ready)
  - Database schema documentation
  - Contribution guidelines enhancement

- [ ] **User Documentation**
  - Video tutorials
  - Feature walkthroughs
  - FAQ section
  - Troubleshooting guide

### Phase 3: Performance & Scalability (Priority: Medium)
**Timeline: Next 4-6 weeks**

- [ ] **Caching**
  - Redis for session storage
  - Query result caching
  - CDN for static assets
  - API response caching

- [ ] **Database Optimization**
  - Index optimization
  - Query performance profiling
  - Connection pooling tuning
  - Read replicas for scaling

- [ ] **Background Jobs**
  - Celery/RQ for async tasks
  - Job queue for molecule generation
  - Batch processing for large datasets
  - Progress tracking and notifications

- [ ] **API Improvements**
  - GraphQL endpoint (optional)
  - WebSocket for real-time updates
  - Streaming for large responses
  - Pagination improvements

### Phase 4: Monitoring & Observability (Priority: Medium)
**Timeline: Next 4-6 weeks**

- [ ] **Logging**
  - Structured logging (JSON format)
  - Log aggregation (ELK stack or Loki)
  - Request tracing
  - Audit logs for security events

- [ ] **Error Tracking**
  - Sentry integration
  - Error grouping and alerts
  - Source map support
  - Performance monitoring

- [ ] **Metrics**
  - Prometheus metrics
  - Grafana dashboards
  - API latency tracking
  - Resource usage alerts
  - Custom business metrics

- [ ] **Health Checks**
  - Liveness and readiness probes
  - Database connectivity checks
  - External service checks
  - Automated alerts

### Phase 5: Advanced Features (Priority: Low)
**Timeline: Next 2-3 months**

- [ ] **API Keys**
  - API key generation
  - Usage tracking per key
  - Rate limiting per key
  - Key rotation

- [ ] **Webhooks**
  - Event-driven notifications
  - Webhook management UI
  - Retry logic
  - Signature verification

- [ ] **Collaboration**
  - Team workspaces
  - Project sharing
  - Comment system
  - Activity feed

- [ ] **Advanced Analytics**
  - Usage statistics dashboard
  - Generation success rates
  - Popular properties/elements
  - User behavior analytics

- [ ] **ML Improvements**
  - Multi-property optimization
  - Active learning pipeline
  - Model versioning
  - A/B testing for models

### Phase 6: Enterprise Features (Priority: Low)
**Timeline: Future**

- [ ] **SSO Integration**
  - SAML support
  - LDAP integration
  - OAuth providers

- [ ] **Compliance**
  - GDPR compliance tools
  - Data export/deletion
  - Audit trail
  - Compliance reporting

- [ ] **Advanced Security**
  - IP whitelisting
  - VPC peering
  - Private endpoints
  - WAF integration

- [ ] **High Availability**
  - Multi-region deployment
  - Automatic failover
  - Disaster recovery
  - 99.9% uptime SLA

---

## Technical Debt

### Immediate
- [ ] Remove duplicate code in models directory (legacy vs current)
- [ ] Standardize error handling across API endpoints
- [ ] Consolidate environment variable handling
- [ ] Update all dependencies to latest stable versions

### Near-term
- [ ] Migrate from SQLite to PostgreSQL for molecule library (for production)
- [ ] Implement proper database migrations (Alembic)
- [ ] Add proper request validation middleware
- [ ] Refactor large components into smaller modules

### Long-term
- [ ] Consider microservices architecture for compute-heavy tasks
- [ ] Evaluate moving to gRPC for internal services
- [ ] Implement event sourcing for audit trail
- [ ] Consider serverless for burst workloads

---

## Performance Targets

### Current
- Page load: ~2-3s
- API latency: ~100-500ms
- Generation time: ~5-10s per molecule
- Concurrent users: ~10-50

### Target (Phase 3)
- Page load: <1s
- API latency: <100ms (cached), <200ms (uncached)
- Generation time: <3s per molecule
- Concurrent users: 500+
- Database queries: <50ms p99

---

## Success Metrics

### Adoption
- Monthly active users
- API requests per day
- Molecules generated
- User retention rate

### Quality
- Test coverage >80%
- Security scan pass rate 100%
- Zero critical vulnerabilities
- API uptime >99.5%

### Performance
- API p95 latency <200ms
- Error rate <0.1%
- Time to first byte <500ms
- Database query time <100ms

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute to these initiatives.

For questions or suggestions, open an issue with the `roadmap` label.
