"""Student research choices. The infrastructure and mode A work before these are filled.

Do not copy a reference answer before recording your own hypothesis and rule.
All scene coordinates currently come from simulator state (oracle condition).
This exercise does not establish image-based accessibility recognition.
"""


def assess_accessibility(scene: dict, proposal: dict) -> dict:
    """Return {'accessible': bool, 'reason': str, 'blockers': list[str]}.

    Available: scene['positions'], ['yaws'], ['home_zones'], ['sizes'],
    ['robot_base'], ['reach_limits']; proposal has object and target_zone.
    Decide what clearance means, how uncertainty enters, and how to test errors.
    """
    raise NotImplementedError('TODO 1: design and justify your accessibility criterion')


def choose_next(scene: dict, proposals: list[dict], assessments: list[dict], history: list[dict]):
    """Return a valid subtask dict or None when no defensible action exists.

    Decide how to select a blocker, whether to skip a proposal, and when to stop.
    You may construct a new transport subtask with vlapipe.planner.to_subtask.
    The framework enforces the same global budget and per-object retry cap.
    """
    raise NotImplementedError('TODO 2: design next-action / blocker selection')


def should_replan(scene: dict, feedback: dict, history: list[dict]) -> bool:
    """Feedback has success, failure, timed_out, lifted, already_satisfied, steps.

    Decide which observations justify replanning. Explain the cost of too many
    model calls and the failure mode of repeating the same unsuccessful action.
    """
    raise NotImplementedError('TODO 3: design the feedback and replanning rule')
