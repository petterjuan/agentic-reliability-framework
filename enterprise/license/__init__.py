# agentic_reliability_framework/enterprise/license/__init__.py
# Foundation
"""
Enterprise License System - Sophisticated license management
"""

from .manager import LicenseManager, LicenseError, LicenseTier, FeatureEntitlement
from .validator import LicenseValidator, validate_license_format
from .crypto import LicenseCrypto, sign_license, verify_license_signature
from .offline import OfflineLicenseValidator, create_offline_license
from .trial import TrialLicenseManager, create_trial_license, validate_trial_license

__all__ = [
    "LicenseManager",
    "LicenseError",
    "LicenseTier",
    "FeatureEntitlement",
    "LicenseValidator",
    "validate_license_format",
    "LicenseCrypto",
    "sign_license",
    "verify_license_signature",
    "OfflineLicenseValidator",
    "create_offline_license",
    "TrialLicenseManager",
    "create_trial_license",
    "validate_trial_license",
]
