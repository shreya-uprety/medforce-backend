"""
Lazy-init shared dependencies used across multiple routers.
"""

import logging

logger = logging.getLogger("medforce-server")

# Global singletons - initialized lazily
chat_agent = None
gcs = None


def get_chat_agent():
    """Lazy initialization of PreConsulteAgent"""
    global chat_agent
    if chat_agent is None:
        try:
            from medforce.agents.pre_consult_agents import PreConsulteAgent
            logger.info("Initializing PreConsulteAgent (lazy)...")
            chat_agent = PreConsulteAgent()
            logger.info("PreConsulteAgent initialized successfully")
        except Exception as e:
            logger.error(f"PreConsulteAgent initialization failed: {e}")
    return chat_agent


def get_gcs():
    """Lazy initialization of GCS Bucket Manager"""
    global gcs
    if gcs is None:
        try:
            from medforce.infrastructure.gcs import GCSBucketManager
            logger.info("Initializing GCS Bucket Manager (lazy)...")
            gcs = GCSBucketManager(bucket_name="clinic_sim_dev")
            logger.info("GCS Bucket Manager initialized successfully")
        except Exception as e:
            logger.error(f"GCS Bucket Manager initialization failed: {e}")
    return gcs


def get_gateway():
    """Get the Gateway singleton (initialized during startup)."""
    try:
        from medforce.gateway.setup import get_gateway as _get_gw
        return _get_gw()
    except Exception as e:
        logger.error(f"Gateway access failed: {e}")
        return None
