# Repository Maintenance

## Essential Workflows
- `v3_comprehensive.yml` - Primary testing workflow
- `release.yml` - Release automation
- `publish.yml` - PyPI publishing
- `oss_tests.yml` - OSS-specific tests

## Essential Scripts
- `smart_v3_validator.py` - V3 architecture validation
- `enforce_oss_purity.py` - OSS boundary enforcement
- `review_artifacts.py` - Release artifact review

## Cleanup Rules
1. Delete workflows older than 2 weeks unless actively used
2. Keep only one version of each workflow type (test, publish, release)
3. Delete one-time fix scripts after 1 week
4. Archive old test files to /archive/ directory
