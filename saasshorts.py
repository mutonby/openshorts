"""Compat shim: re-exports openshorts.saas.pipeline at the original import path.

The SaaS UGC pipeline moved to openshorts/saas/pipeline.py as part of the
restructure. A future commit may split it further into research / scripting /
media / compositing / pipeline modules per the plan; for now it lives as a
single module in the saas/ folder. New code should import from
`openshorts.saas.pipeline` directly; this shim keeps existing
`from saasshorts import ...` calls working.
"""
from openshorts.saas.pipeline import *  # noqa: F401,F403
from openshorts.saas.pipeline import (  # noqa: F401
    scrape_website,
    research_saas_online,
    analyze_saas,
    generate_scripts,
    generate_full_video,
    generate_actor_images,
    generate_actor_image,
    generate_voiceover,
    get_elevenlabs_voices,
    DEFAULT_VOICES,
)
