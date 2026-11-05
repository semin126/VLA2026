"""Final project ver2.0: image-based strategy, VLA execution and paired trials.

Version 1 research_lab and student_design remain independent and unchanged.
"""

PROTOCOL = 'final_project_v2.0'

# policy -> controller imports MuJoCo before env.py can choose EGL. Configure
# at package entry so CLI/data/validation paths select the installed GPU driver
# before that first import. This module uses only standard-library dependencies.
from ..render_setup import configure_headless_rendering

RENDER_SETUP = configure_headless_rendering()
