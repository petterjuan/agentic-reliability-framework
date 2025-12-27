# agentic_reliability_framework/app.py (OSS-ONLY VERSION)
"""
Agentic Reliability Framework - OSS Edition DEMO
Demo application showing OSS advisory capabilities

NOTES:
- OSS edition: Advisory only, no execution
- HealingIntent creation only, no execution
- In-memory storage only, no persistence
"""

import asyncio
import datetime
import logging
import sys
from typing import Optional, Dict, Any

import gradio as gr
import numpy as np

# Import OSS components directly
from agentic_reliability_framework.arf_core.models.healing_intent import (
    HealingIntent,
    create_rollback_intent,
    create_restart_intent,
    create_scale_out_intent,
)
from agentic_reliability_framework.arf_core.engine.simple_mcp_client import OSSMCPClient

logger = logging.getLogger(__name__)

# ============================================================================
# DEMO SCENARIOS FOR OSS EDITION
# ============================================================================

DEMO_SCENARIOS = {
    "🚀 OSS Demo - Latency Spike": {
        "component": "api-service",
        "latency": 250.0,
        "error_rate": 0.15,
        "story": """
**OSS ADVISORY DEMO: Latency Spike Detection**

This is the OSS edition showing advisory capabilities only.

🔍 **What OSS detects:**
- Latency spike from 85ms to 250ms (294% increase)
- Error rate at 15% (above 5% threshold)
- Pattern matches historical incidents

🤖 **What OSS recommends:**
- Create HealingIntent for container restart
- Confidence: 85% (based on similar incidents)
- Advisory only - requires Enterprise for execution

💼 **Enterprise Upgrade Benefits:**
- Autonomous execution of HealingIntent
- Approval workflows for safety
- Learning engine for continuous improvement
- Persistent storage for incident history
        """
    },
    
    "📈 OSS Demo - Error Rate Increase": {
        "component": "database-service", 
        "latency": 120.0,
        "error_rate": 0.25,
        "story": """
**OSS ADVISORY DEMO: Error Rate Analysis**

OSS edition analyzes patterns and creates HealingIntent.

🔍 **OSS Analysis:**
- Error rate at 25% (critical threshold: 15%)
- Moderate latency increase
- Database connection pattern detected

🤖 **OSS Recommendation:**
- Create HealingIntent for rollback to previous version
- Confidence: 78% (based on RAG similarity)
- Advisory analysis complete

🔒 **OSS Safety:**
- No execution capability (safety feature)
- Creates intent for Enterprise execution
- Full audit trail when upgraded
        """
    },
}

# ============================================================================
# OSS-ONLY APPLICATION LOGIC
# ============================================================================

async def analyze_with_oss(
    component: str,
    latency: float,
    error_rate: float,
    scenario_name: str = "OSS Demo"
) -> Dict[str, Any]:
    """
    OSS-only analysis: Creates HealingIntent, never executes
    
    This is the core OSS capability: analyze and recommend.
    """
    try:
        # Create OSS MCP client (advisory only)
        client = OSSMCPClient()
        
        # Based on metrics, decide which HealingIntent to create
        healing_intent = None
        
        if error_rate > 0.2:
            # High error rate → recommend rollback
            healing_intent = create_rollback_intent(
                component=component,
                revision="previous",
                justification=f"High error rate ({error_rate*100:.1f}%) detected in {component}",
                incident_id=f"oss_demo_{datetime.datetime.now().timestamp()}"
            )
            action = "rollback"
        elif latency > 200:
            # High latency → recommend restart
            healing_intent = create_restart_intent(
                component=component,
                justification=f"High latency ({latency:.0f}ms) detected in {component}",
                incident_id=f"oss_demo_{datetime.datetime.now().timestamp()}"
            )
            action = "restart_container"
        else:
            # Scale out recommendation
            healing_intent = create_scale_out_intent(
                component=component,
                scale_factor=2,
                justification=f"Performance degradation in {component}",
                incident_id=f"oss_demo_{datetime.datetime.now().timestamp()}"
            )
            action = "scale_out"
        
        # Get OSS MCP analysis (advisory only)
        mcp_result = await client.execute_tool({
            "tool": action,
            "component": component,
            "parameters": {},
            "justification": healing_intent.justification,
            "metadata": {
                "scenario": scenario_name,
                "latency": latency,
                "error_rate": error_rate,
                "oss_edition": True
            }
        })
        
        return {
            "status": "OSS_ADVISORY_COMPLETE",
            "healing_intent": healing_intent.to_enterprise_request(),
            "oss_analysis": mcp_result,
            "confidence": healing_intent.confidence,
            "requires_enterprise": True,
            "message": f"✅ OSS analysis complete. Created HealingIntent for {action} on {component}.",
            "enterprise_upgrade_url": "https://arf.dev/enterprise",
            "enterprise_features": [
                "Autonomous execution",
                "Approval workflows", 
                "Learning engine",
                "Persistent storage",
                "Audit trails",
                "Compliance reporting"
            ]
        }
        
    except Exception as e:
        logger.error(f"OSS analysis error: {e}", exc_info=True)
        return {
            "status": "OSS_ERROR",
            "message": f"❌ OSS analysis failed: {str(e)}",
            "requires_enterprise": False
        }

# ============================================================================
# SIMPLE OSS DEMO UI
# ============================================================================

def create_oss_demo_ui():
    """Create simple OSS demo UI"""
    with gr.Blocks(title="🧠 ARF OSS Edition - Advisory Demo", theme="soft") as demo:
        gr.Markdown("""
        # 🧠 Agentic Reliability Framework - OSS Edition
        **Advisory Analysis Only - Apache 2.0 Licensed**
        
        _OSS creates HealingIntent recommendations. Enterprise executes them._
        
        ---
        
        **OSS Capabilities:**
        ✅ Anomaly detection & pattern recognition  
        ✅ HealingIntent creation (advisory only)  
        ✅ RAG similarity search (in-memory)  
        ✅ Safety validation & guardrails  
        
        **Enterprise Upgrade Required For:**
        🔧 Actual tool execution  
        📊 Learning from outcomes  
        💾 Persistent storage  
        📝 Audit trails & compliance  
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Demo Input")
                
                scenario = gr.Dropdown(
                    choices=list(DEMO_SCENARIOS.keys()),
                    value="🚀 OSS Demo - Latency Spike",
                    label="Demo Scenario"
                )
                
                component = gr.Textbox(
                    value="api-service",
                    label="Component",
                    interactive=True
                )
                
                latency = gr.Slider(
                    minimum=10, maximum=1000, value=250,
                    label="Latency (ms)",
                    info="P99 latency in milliseconds"
                )
                
                error_rate = gr.Slider(
                    minimum=0, maximum=1, value=0.15, step=0.01,
                    label="Error Rate",
                    info="Error rate (0.0 to 1.0)"
                )
                
                analyze_btn = gr.Button("🤖 Analyze with OSS", variant="primary")
            
            with gr.Column(scale=2):
                gr.Markdown("### 📋 OSS Analysis Results")
                
                scenario_story = gr.Markdown(
                    value=DEMO_SCENARIOS["🚀 OSS Demo - Latency Spike"]["story"]
                )
                
                output = gr.JSON(
                    label="OSS Analysis Output",
                    value={}
                )
        
        def update_scenario(scenario_name):
            """Update UI when scenario changes"""
            scenario_data = DEMO_SCENARIOS.get(scenario_name, {})
            return {
                scenario_story: gr.update(value=scenario_data.get("story", "")),
                component: gr.update(value=scenario_data.get("component", "api-service")),
                latency: gr.update(value=scenario_data.get("latency", 100)),
                error_rate: gr.update(value=scenario_data.get("error_rate", 0.05)),
            }
        
        async def analyze_oss_async(component, latency, error_rate, scenario_name):
            """Async analysis handler"""
            result = await analyze_with_oss(component, latency, error_rate, scenario_name)
            return result
        
        # Connect events
        scenario.change(
            fn=update_scenario,
            inputs=[scenario],
            outputs=[scenario_story, component, latency, error_rate]
        )
        
        analyze_btn.click(
            fn=analyze_oss_async,
            inputs=[component, latency, error_rate, scenario],
            outputs=[output]
        )
        
        # Footer with upgrade info
        gr.Markdown("""
        ---
        
        **Upgrade to Enterprise Edition:**
        
        | Feature | OSS Edition | Enterprise Edition |
        |---------|-------------|-------------------|
        | Execution | ❌ Advisory only | ✅ Autonomous + Approval |
        | Storage | ⚠️ In-memory only | ✅ Persistent (Neo4j, PostgreSQL) |
        | Learning | ❌ None | ✅ Continuous learning engine |
        | Audit | ❌ None | ✅ Full audit trails (SOC2, HIPAA) |
        | Support | ❌ Community | ✅ 24/7 Enterprise support |
        
        **Contact:** enterprise@petterjuan.com | **Website:** https://arf.dev
        """)
    
    return demo

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for OSS demo"""
    logging.basicConfig(level=logging.INFO)
    logger.info("=" * 60)
    logger.info("Starting ARF OSS Edition - Advisory Demo")
    logger.info("=" * 60)
    
    demo = create_oss_demo_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
