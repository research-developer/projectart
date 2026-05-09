"""Tracking — TrackedEntity hierarchy + smart container.

YOLO detections come in; the registry routes them to existing entities or
instantiates new ones. Entities are the long-lived state; behaviors are
the rule layer that subscribes to entity lifecycle events.

Public API:
    from projectart.tracking import TrackedEntity, TrackedRegistry, EntityState
    from projectart.tracking.builtins import Cat, Person
"""
from .entity import BBox, EntityState, TrackedEntity
from .registry import TrackedRegistry

__all__ = ["BBox", "EntityState", "TrackedEntity", "TrackedRegistry"]
