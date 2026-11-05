"""Mechanical screening for human demonstrations; this is not human review.

No input is relabelled as successful simply because the operator pressed Save.
The recorded observations/actions remain unchanged. A reviewer must check task
semantics and visible collisions before using candidates for training.
"""
from __future__ import annotations

import uuid
from pathlib import Path
import numpy as np
from . import config as C


def screen_recording(writer, env, object_name=None, target_zone=None):
    """Require a newly completed, lifted, released, supported, settled transfer."""
    n = len(writer)
    report = {"frames": n, "success": False, "screen": "mechanical_candidate_v1",
              "review_required": True, "review_status": "pending", "problems": []}
    if n < 5:
        report["problems"].append("fewer_than_5_actions")
        return report
    q = np.asarray(writer.qpos)
    actions = np.asarray(writer.actions)
    if actions.shape != (n, C.ACTION_DIM) or not np.isfinite(actions).all():
        report["problems"].append("invalid_actions")
    grip = np.asarray([o["robot0_gripper_qpos"] for o in writer.obs])
    released = bool(np.all(grip[-5:].min(axis=1) > 0.035))
    report["released_last_5_frames"] = released
    if not released:
        report["problems"].append("gripper_not_released")
    names = [object_name] if object_name else list(C.OBJECT_NAMES)
    moved = []
    outcomes = []
    for name in names:
        adr = int(env._jnt_adr[name])
        xyz = q[:, adr:adr + 3]
        half_z = C.half_size(C.OBJECT_BY_NAME[name])[2]
        goal = target_zone if object_name else (C.OBJECT_BY_NAME[name]["home_zone"]
                    if writer.direction == "tidy" else "center_zone")
        zone = C.ZONES[goal]
        def inside(p):
            return bool(zone["x"][0] <= p[0] <= zone["x"][1]
                        and zone["y"][0] <= p[1] <= zone["y"][1])
        initially_done = inside(xyz[0])
        completed = inside(xyz[-1])
        lifted = bool(xyz[:, 2].max() > C.TABLE_TOP_Z + half_z + 0.04)
        supported = bool(abs(float(xyz[-1, 2]) - C.TABLE_TOP_Z - half_z) < 0.02)
        speed = float(np.linalg.norm(np.diff(xyz[-6:], axis=0), axis=1).max() * C.CONTROL_HZ)
        changed = not initially_done
        if changed:
            moved.append(name)
            if not (completed and lifted and supported and speed < 0.05):
                report["problems"].append(f"{name}:goal_lift_support_or_settle_failed")
        elif not completed or not supported:
            report["problems"].append(f"{name}:initially_correct_object_disturbed")
        outcomes.append({"object": name, "target_zone": goal, "initially_satisfied": initially_done,
                         "goal_reached": completed, "lifted": lifted, "supported": supported,
                         "final_max_speed_m_s": speed})
    if not moved:
        report["problems"].append("no_new_goal_completed")
    report.update({"outcomes": outcomes, "new_goal_objects": moved,
                   "success": not report["problems"]})
    return report


def save_recording(writer, env, root: Path, object_name=None, target_zone=None, force_reject_reason=None):
    """Preserve both candidate successes and failures in fresh files."""
    report = screen_recording(writer, env, object_name, target_zone)
    if force_reject_reason:
        report["problems"].append(force_reject_reason)
        report["success"] = False
    writer.meta.update(report)
    bucket = "candidates" if report["success"] else "rejected"
    path = Path(root) / bucket / ("ep_" + uuid.uuid4().hex + ".hdf5")
    if path.exists():
        raise FileExistsError(path)
    # Zero-action recordings cannot form the episode format. Save their report only.
    if len(writer):
        writer.save(path)
    else:
        path = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
    import json
    path.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, report
