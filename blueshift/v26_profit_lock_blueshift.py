"""Blueshift baseline strategy entry for v26-profit-lock."""

from blueshift_library.orb_v26_runtime import handle_data, initialize_strategy


def initialize(context):
    initialize_strategy(
        context,
        overrides={
            "script_version": "v26-profit-lock-blueshift",
            "baseline_reference": "v26-profit-lock",
            "research_only": False,
            "orb_reentry_enabled": False,
        },
    )

