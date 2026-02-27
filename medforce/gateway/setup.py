"""
Gateway Setup — initializes and wires together all Gateway components.

Called once during app startup. If anything fails, the Gateway is disabled
and the rest of the app continues working normally.
"""

from __future__ import annotations

import logging

from medforce.gateway.agents.booking_agent import BookingAgent
from medforce.gateway.booking_registry import BookingRegistry
from medforce.gateway.agents.clinical_agent import ClinicalAgent
from medforce.gateway.agents.intake_agent import IntakeAgent
from medforce.gateway.agents.monitoring_agent import MonitoringAgent
from medforce.gateway.handlers.gp_comms import GPCommunicationHandler
from medforce.gateway.heartbeat import HeartbeatScheduler
from medforce.gateway.channels import DispatcherRegistry
from medforce.gateway.diary import DiaryStore
from medforce.gateway.dispatchers.test_harness_dispatcher import (
    TestHarnessDispatcher,
)
from medforce.gateway.dispatchers.websocket_dispatcher import (
    WebSocketDispatcher,
)
from medforce.gateway.gateway import Gateway
from medforce.gateway.handlers.identity_resolver import IdentityResolver
from medforce.gateway.permissions import PermissionChecker
from medforce.gateway.queue import PatientQueueManager

logger = logging.getLogger("gateway.setup")

# Module-level singletons (set during initialize)
_gateway: Gateway | None = None
_queue_manager: PatientQueueManager | None = None
_dispatcher_registry: DispatcherRegistry | None = None
_identity_resolver: IdentityResolver | None = None
_diary_store: DiaryStore | None = None
_heartbeat_scheduler: HeartbeatScheduler | None = None


async def initialize_gateway() -> Gateway:
    """
    Wire together all Gateway components and start background tasks.

    Returns the fully initialized Gateway instance.
    """
    global _gateway, _queue_manager, _dispatcher_registry
    global _identity_resolver, _diary_store, _heartbeat_scheduler

    logger.info("Initializing MedForce Gateway...")

    # 1. GCS-backed diary store (eager init to avoid cold-start on first request)
    from medforce.dependencies import get_gcs
    gcs = get_gcs()
    gcs._ensure_initialized()
    _diary_store = DiaryStore(gcs)

    # 2. Dispatcher registry
    _dispatcher_registry = DispatcherRegistry()
    _dispatcher_registry.register(WebSocketDispatcher())
    _dispatcher_registry.register(TestHarnessDispatcher())

    # Phase 6: Register external channel dispatchers (conditional on config)
    _register_external_dispatchers(_dispatcher_registry)

    # 3. Identity resolver
    _identity_resolver = IdentityResolver()

    # 4. Permission checker
    permission_checker = PermissionChecker()

    # 5. Gateway
    _gateway = Gateway(
        diary_store=_diary_store,
        dispatcher_registry=_dispatcher_registry,
        permission_checker=permission_checker,
    )

    # 6. Register agents
    # Booking registry for slot holds / double-booking prevention
    booking_registry = BookingRegistry(gcs_bucket_manager=gcs)

    _gateway.register_agent("intake", IntakeAgent(gcs_bucket_manager=gcs))
    _gateway.register_agent("clinical", ClinicalAgent())
    _gateway.register_agent("booking", BookingAgent(booking_registry=booking_registry))
    _gateway.register_agent("gp_comms", GPCommunicationHandler())
    _gateway.register_agent("monitoring", MonitoringAgent())

    # 7. Queue manager (uses gateway.process_event as the processor)
    _queue_manager = PatientQueueManager(processor=_gateway.process_event)
    await _queue_manager.start()

    # 8. Heartbeat scheduler (fires HEARTBEAT events for monitored patients)
    _heartbeat_scheduler = HeartbeatScheduler(
        processor=_gateway.process_event,
        diary_store=_diary_store,
    )
    await _heartbeat_scheduler.start()

    logger.info(
        "Gateway initialized: agents=%s, channels=%s",
        _gateway.registered_agents,
        _dispatcher_registry.registered_channels,
    )

    return _gateway


async def shutdown_gateway() -> None:
    """Gracefully stop background tasks."""
    global _queue_manager, _heartbeat_scheduler
    if _heartbeat_scheduler:
        await _heartbeat_scheduler.stop()
    if _queue_manager:
        await _queue_manager.stop()
        logger.info("Gateway shutdown complete")


def get_gateway() -> Gateway | None:
    return _gateway


def get_queue_manager() -> PatientQueueManager | None:
    return _queue_manager


def get_dispatcher_registry() -> DispatcherRegistry | None:
    return _dispatcher_registry


def get_identity_resolver() -> IdentityResolver | None:
    return _identity_resolver


def get_diary_store() -> DiaryStore | None:
    return _diary_store


def get_heartbeat_scheduler() -> HeartbeatScheduler | None:
    return _heartbeat_scheduler


def _register_external_dispatchers(registry: DispatcherRegistry) -> None:
    """
    Register Phase 6 channel dispatchers based on environment variables.

    Each dispatcher is independently conditional — you can enable any
    combination of Dialogflow, Email, and Twilio.
    """
    import os

    # Dialogflow CX → WhatsApp/SMS
    if os.getenv("DIALOGFLOW_PROJECT_ID"):
        try:
            from medforce.gateway.dispatchers.dialogflow_dispatcher import (
                DialogflowDispatcher,
            )
            registry.register(DialogflowDispatcher())
            logger.info("Dialogflow dispatcher registered")
        except Exception as exc:
            logger.warning("Dialogflow dispatcher failed to register: %s", exc)

    # Email via SendGrid (for GP communication)
    if os.getenv("SENDGRID_API_KEY"):
        try:
            from medforce.gateway.dispatchers.email_dispatcher import (
                EmailDispatcher,
            )
            registry.register(EmailDispatcher())
            logger.info("Email dispatcher registered")
        except Exception as exc:
            logger.warning("Email dispatcher failed to register: %s", exc)

    # Twilio SMS (alternative/complement to Dialogflow)
    if os.getenv("TWILIO_ACCOUNT_SID"):
        try:
            from medforce.gateway.dispatchers.twilio_dispatcher import (
                TwilioSMSDispatcher,
            )
            registry.register(TwilioSMSDispatcher())
            logger.info("Twilio SMS dispatcher registered")
        except Exception as exc:
            logger.warning("Twilio SMS dispatcher failed to register: %s", exc)
