"""Domain model compatibility adapters."""

from mm.core.adapters.joint import (
    action_to_joint_target_book,
    action_to_venue_target_book,
    build_frame_sync_status,
    joint_ledger_from_frame,
    joint_strategy_frame_from_legacy,
    joint_target_book_to_action,
    strategy_input_frame_to_joint_strategy_frame,
    venue_ledger_from_private_snapshot,
    venue_target_book_to_action,
)

__all__ = [
    "action_to_joint_target_book",
    "action_to_venue_target_book",
    "build_frame_sync_status",
    "joint_ledger_from_frame",
    "joint_strategy_frame_from_legacy",
    "joint_target_book_to_action",
    "strategy_input_frame_to_joint_strategy_frame",
    "venue_ledger_from_private_snapshot",
    "venue_target_book_to_action",
]
