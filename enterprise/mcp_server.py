# agentic_reliability_framework/enterprise/mcp_server.py
"""
Enterprise MCP Server - Enhanced with license validation and Enterprise features
"""

import os
import logging
from typing import Dict, Any, Optional

from ...engine.mcp_server import MCPServer, MCPMode, MCPRequest, MCPResponse
from ...engine.mcp_factory import detect_edition
from ..license.manager import LicenseManager, LicenseError, FeatureEntitlement

logger = logging.getLogger(__name__)


class EnterpriseMCPServer(MCPServer):
    """
    Enterprise MCP Server - Enhanced version of MCPServer
    
    Key Enterprise features:
    1. License validation with feature entitlements
    2. HealingIntent execution from OSS
    3. Enhanced audit trails
    4. Learning engine integration
    5. Persistent storage integration
    """
    
    def __init__(
        self,
        mode: Optional[MCPMode] = None,
        license_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Enterprise MCP Server
        
        Args:
            mode: MCP execution mode
            license_key: Enterprise license key
            config: Configuration dictionary
        """
        # Initialize license manager
        self.license_manager = LicenseManager(license_key)
        
        # Validate license before proceeding
        if not self.license_manager.validate():
            raise LicenseError(
                f"Valid Enterprise license required. "
                f"Get a trial at https://arf.dev/trial\n"
                f"Current license: {license_key[:20] if license_key else 'None'}"
            )
        
        # Get entitlements
        entitlements = self.license_manager.get_entitlements()
        if not entitlements:
            raise LicenseError("No valid license entitlements found")
        
        # Determine default mode based on license tier
        if mode is None:
            if entitlements.tier.value == "team":
                mode = MCPMode.APPROVAL
            elif entitlements.tier.value in ["business", "enterprise"]:
                mode = MCPMode.AUTONOMOUS
            else:
                mode = MCPMode.ADVISORY
        
        # Check if mode is allowed by license
        if not self.license_manager.can_execute(mode.value):
            raise LicenseError(
                f"License tier '{entitlements.tier.value}' does not allow '{mode.value}' mode. "
                f"Allowed modes: {self._get_allowed_modes(entitlements)}"
            )
        
        # Store Enterprise context
        self.license_info = self.license_manager.get_info()
        self.entitlements = entitlements
        
        # Initialize parent MCPServer with validated mode
        super().__init__(mode=mode)
        
        # Enterprise enhancements
        self.audit_trail = []
        self.enterprise_config = config or {}
        
        logger.info(f"🚀 Enterprise MCPServer initialized")
        logger.info(f"📋 License: {self.license_info}")
        logger.info(f"🎯 Mode: {self.mode.value}")
        logger.info(f"💼 Tier: {entitlements.tier.value}")
    
    def _get_allowed_modes(self, entitlements: FeatureEntitlement) -> list:
        """Get allowed modes based on entitlements"""
        modes = ["advisory"]  # Always allowed
        
        if entitlements.has_feature("approval"):
            modes.append("approval")
        
        if entitlements.has_feature("autonomous"):
            modes.append("autonomous")
        
        return modes
    
    async def execute_tool(self, request_dict: Dict[str, Any]) -> MCPResponse:
        """
        Enterprise-enhanced execute_tool with license validation
        
        Maintains exact same API signature as OSS version
        """
        # Check license status before execution
        if not self.license_manager.validate():
            if self.license_manager.get_info().get("in_grace_period"):
                logger.warning(f"License expired, in grace period. Features limited.")
                # Allow only advisory mode in grace period
                request_dict["mode"] = "advisory"
            else:
                raise LicenseError("License expired or invalid")
        
        # Add license context to request
        enhanced_request = self._add_license_context(request_dict)
        
        # Check if this is an OSS HealingIntent
        metadata = enhanced_request.get("metadata", {})
        is_oss_intent = metadata.get("requires_enterprise", False)
        
        if is_oss_intent:
            # Execute with Enterprise enhancements
            return await self._execute_enterprise_intent(enhanced_request)
        
        # Standard execution with Enterprise features
        return await super().execute_tool(enhanced_request)
    
    def _add_license_context(self, request_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add license context to request"""
        metadata = request_dict.get("metadata", {})
        
        return {
            **request_dict,
            "metadata": {
                **metadata,
                "license_tier": self.entitlements.tier.value,
                "license_id": self.license_info.get("id", "unknown"),
                "entitlements": self.entitlements.to_dict(),
                "enterprise_features": True,
            }
        }
    
    async def _execute_enterprise_intent(self, request_dict: Dict[str, Any]) -> MCPResponse:
        """Execute OSS-generated HealingIntent with Enterprise features"""
        # Start audit trail
        audit_id = self._start_audit_trail(request_dict)
        
        try:
            # Import HealingIntent
            from ...oss.healing_intent import HealingIntent
            
            # Convert to HealingIntent
            intent = HealingIntent.from_mcp_request(request_dict)
            
            # Log intent execution
            self._log_intent_execution(intent, audit_id)
            
            # Create enhanced request
            enterprise_request = {
                **request_dict,
                "mode": self.mode.value,
                "metadata": {
                    **request_dict.get("metadata", {}),
                    "audit_id": audit_id,
                    "intent_id": intent.deterministic_id,
                    "enterprise_executed": True,
                }
            }
            
            # Execute using parent class
            response = await super().execute_tool(enterprise_request)
            
            # Complete audit trail
            self._complete_audit_trail(audit_id, response)
            
            return response
            
        except Exception as e:
            # Log failure in audit trail
            self._fail_audit_trail(audit_id, str(e))
            raise
    
    def _start_audit_trail(self, request: Dict[str, Any]) -> str:
        """Start audit trail entry"""
        import time
        audit_id = f"audit_{int(time.time())}_{len(self.audit_trail)}"
        
        entry = {
            "audit_id": audit_id,
            "timestamp": time.time(),
            "action": "start_execution",
            "request": {
                "tool": request.get("tool"),
                "component": request.get("component"),
                "mode": request.get("mode"),
            },
            "license": self.license_info,
        }
        
        self.audit_trail.append(entry)
        return audit_id
    
    def _log_intent_execution(self, intent, audit_id: str):
        """Log intent execution"""
        import time
        entry = {
            "audit_id": audit_id,
            "timestamp": time.time(),
            "action": "execute_intent",
            "intent_id": intent.deterministic_id,
            "tool": intent.action,
            "component": intent.component,
