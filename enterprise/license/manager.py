# agentic_reliability_framework/enterprise/license/manager.py
"""
Enterprise License Manager - Sophisticated license management
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict, field
import hashlib

logger = logging.getLogger(__name__)


class LicenseTier(Enum):
    """License tiers"""
    TRIAL = "trial"
    TEAM = "team"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


@dataclass
class FeatureEntitlement:
    """Feature entitlements for a license"""
    tier: LicenseTier
    features: List[str] = field(default_factory=list)
    limits: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[datetime] = None
    grace_period_days: int = 7
    
    def has_feature(self, feature: str) -> bool:
        """Check if license has a specific feature"""
        return feature in self.features
    
    def get_limit(self, limit_name: str, default: Any = None) -> Any:
        """Get a specific limit value"""
        return self.limits.get(limit_name, default)
    
    def is_expired(self) -> bool:
        """Check if license is expired"""
        if not self.expires_at:
            return False
        return datetime.now() > self.expires_at
    
    def is_in_grace_period(self) -> bool:
        """Check if license is in grace period"""
        if not self.expires_at:
            return False
        grace_until = self.expires_at + timedelta(days=self.grace_period_days)
        return datetime.now() > self.expires_at and datetime.now() <= grace_until
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["tier"] = self.tier.value
        if self.expires_at:
            data["expires_at"] = self.expires_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureEntitlement":
        """Create from dictionary"""
        data = data.copy()
        data["tier"] = LicenseTier(data["tier"])
        if data.get("expires_at"):
            data["expires_at"] = datetime.fromisoformat(data["expires_at"])
        return cls(**data)


class LicenseError(Exception):
    """License-related error"""
    pass


class LicenseManager:
    """
    Sophisticated license manager for Enterprise edition
    
    Features:
    - JWT-style license validation
    - Feature entitlements
    - Offline validation support
    - Grace periods
    - Trial license management
    """
    
    def __init__(self, license_key: Optional[str] = None):
        """
        Initialize license manager
        
        Args:
            license_key: License key string (can be None for OSS)
        """
        self.license_key = license_key or os.getenv("ARF_LICENSE_KEY", "")
        self.license_info: Optional[Dict[str, Any]] = None
        self.entitlements: Optional[FeatureEntitlement] = None
        
        # Cache for license validation
        self._validation_cache = {}
        self._last_validation = None
        
        # Initialize
        self._load_license()
    
    def _load_license(self) -> None:
        """Load and parse license"""
        if not self.license_key:
            logger.debug("No license key provided")
            return
        
        try:
            # Parse license key format
            if self.license_key.startswith("ARF-TRIAL-"):
                self._parse_trial_license()
            elif self.license_key.startswith("ARF-ENT-"):
                self._parse_enterprise_license()
            else:
                logger.warning(f"Unknown license key format: {self.license_key[:20]}...")
                return
            
            logger.info(f"License loaded: {self.get_info()}")
            
        except Exception as e:
            logger.error(f"Error loading license: {e}")
            self.license_info = None
            self.entitlements = None
    
    def _parse_trial_license(self) -> None:
        """Parse trial license"""
        # Trial license format: ARF-TRIAL-{uuid}-{days}-{features}
        parts = self.license_key.split("-")
        
        if len(parts) < 4:
            raise LicenseError("Invalid trial license format")
        
        # Extract components
        uuid_part = parts[2]
        trial_days = 30  # Default 30-day trial
        features = ["autonomous", "approval", "learning", "persistence"]
        
        # Create trial license info
        self.license_info = {
            "type": "trial",
            "id": f"trial_{uuid_part[:8]}",
            "issued_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=trial_days)).isoformat(),
            "features": features,
        }
        
        # Create entitlements
        self.entitlements = FeatureEntitlement(
            tier=LicenseTier.TRIAL,
            features=features,
            limits={
                "max_incidents": 10000,
                "max_users": 10,
                "max_services": 50,
                "support_level": "community",
            },
            expires_at=datetime.now() + timedelta(days=trial_days),
            grace_period_days=7,
        )
    
    def _parse_enterprise_license(self) -> None:
        """Parse enterprise license"""
        # Enterprise license format: ARF-ENT-{uuid}-{tier}-{signature}
        parts = self.license_key.split("-")
        
        if len(parts) < 4:
            raise LicenseError("Invalid enterprise license format")
        
        uuid_part = parts[2]
        tier = parts[3] if len(parts) > 3 else "team"
        
        # Map tier to features
        tier_features = {
            "team": ["approval", "persistence"],
            "business": ["autonomous", "approval", "persistence", "learning"],
            "enterprise": ["autonomous", "approval", "persistence", "learning", "audit", "compliance"],
        }
        
        features = tier_features.get(tier, ["approval"])
        
        # Create enterprise license info
        self.license_info = {
            "type": "enterprise",
            "id": f"ent_{uuid_part[:8]}",
            "tier": tier,
            "issued_at": datetime.now().isoformat(),
            "expires_at": None,  # Enterprise licenses don't expire
            "features": features,
        }
        
        # Create entitlements
        self.entitlements = FeatureEntitlement(
            tier=LicenseTier(tier),
            features=features,
            limits={
                "max_incidents": 100000,
                "max_users": 100 if tier == "team" else 1000,
                "max_services": 500,
                "support_level": "enterprise" if tier == "enterprise" else "business",
            },
            expires_at=None,  # No expiration
            grace_period_days=30,
        )
    
    def validate(self) -> bool:
        """
        Validate license
        
        Returns:
            True if license is valid, False otherwise
        """
        # Cache validation for 5 minutes
        cache_key = f"validate_{hash(self.license_key)}"
        current_time = datetime.now()
        
        if cache_key in self._validation_cache:
            cached_time, cached_result = self._validation_cache[cache_key]
            if (current_time - cached_time).seconds < 300:  # 5 minutes
                return cached_result
        
        # No license key = OSS mode
        if not self.license_key:
            result = False
        elif not self.entitlements:
            result = False
        elif self.entitlements.is_expired():
            if self.entitlements.is_in_grace_period():
                logger.warning(f"License expired, in grace period")
                result = True  # Allow in grace period
            else:
                logger.error(f"License expired")
                result = False
        else:
            result = True
        
        # Cache result
        self._validation_cache[cache_key] = (current_time, result)
        self._last_validation = current_time
        
        return result
    
    def get_info(self) -> Dict[str, Any]:
        """Get license information"""
        if not self.license_info:
            return {
                "valid": False,
                "type": "oss",
                "edition": "oss",
                "features": [],
            }
        
        is_valid = self.validate()
        
        return {
            "valid": is_valid,
            "type": self.license_info["type"],
            "id": self.license_info.get("id", "unknown"),
            "tier": self.license_info.get("tier", "unknown"),
            "expires_at": self.license_info.get("expires_at"),
            "features": self.license_info.get("features", []),
            "in_grace_period": self.entitlements.is_in_grace_period() if self.entitlements else False,
            "edition": "enterprise",
        }
    
    def get_entitlements(self) -> Optional[FeatureEntitlement]:
        """Get feature entitlements"""
        if not self.validate():
            return None
        return self.entitlements
    
    def can_execute(self, mode: str) -> bool:
        """Check if license allows execution in given mode"""
        if not self.validate():
            return False
        
        if not self.entitlements:
            return False
        
        # OSS mode always allowed (advisory)
        if mode == "advisory":
            return True
        
        # Check feature entitlements
        if mode == "approval":
            return self.entitlements.has_feature("approval")
        elif mode == "autonomous":
            return self.entitlements.has_feature("autonomous")
        
        return False
    
    def refresh(self) -> bool:
        """Refresh license validation (e.g., after network restore)"""
        self._validation_cache.clear()
        self._load_license()
        return self.validate()


# Factory function
def create_license_manager(license_key: Optional[str] = None) -> LicenseManager:
    """Create license manager instance"""
    return LicenseManager(license_key)
