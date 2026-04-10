"""Blueshift research-only evaluator for the structural ORB re-entry branch."""

from blueshift_library.orb_v26_runtime import handle_data, initialize_strategy


def initialize(context):
    initialize_strategy(
        context,
        overrides={
            "script_version": "v26-orb-reentry-evaluator-blueshift",
            "baseline_reference": "v26-profit-lock",
            "research_only": True,
            "orb_reentry_enabled": True,
            "orb_reentry_arm_progress_mult": 1.0,
            "orb_reentry_depth_mult": 0.25,
            "orb_reentry_confirm_bars": 1,
        },
    )
