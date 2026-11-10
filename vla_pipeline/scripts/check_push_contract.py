"""CPU-only push/parser/runner regression. No checkpoints or GPU rendering.

Real MuJoCo integration, synthetic blank images and idle model adapters test
dispatch/limits/feedback, not physical push success or learned performance.
Run: python scripts/check_push_contract.py
"""
import _bootstrap  # noqa: F401
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vlapipe import config as C, numbered
from vlapipe.env import SortingEnv
from vlapipe.planner import PUSH_DIRECTIONS, push_destination, push_instruction
from vlapipe.policy import ScriptedSortingPolicy
from vlapipe.final_project_v2.common import dump, parse_response, public_task, task_text
from vlapipe.final_project_v2 import runner
from vlapipe.final_project_v2.scenes import freeze, template

ROOT = Path(__file__).resolve().parents[1]


class IdlePolicy:
    def __init__(self):
        self.tasks = []

    def reset(self):
        pass

    def select_action(self, observation):
        self.tasks.append(observation['task'])
        return np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


class Planner:
    def __init__(self, plan):
        self.plan = plan

    def generate(self, image, prompt, deadline):
        return json.dumps({'assessments': [], 'plan': self.plan, 'done': False})


class PushChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        numbered.activate(count=10)

    def test_labels_and_axes(self):
        suffixes = ['toward the robot', 'away from the robot', 'to the left', 'to the right']
        for direction, suffix in zip(PUSH_DIRECTIONS, suffixes):
            self.assertEqual(push_instruction('object_3', direction), f'Push object 3 {suffix}.')
            displacement = np.asarray(push_destination((.1, .3), direction)) - [.1, .3]
            np.testing.assert_allclose(displacement, np.asarray(PUSH_DIRECTIONS[direction]) * .07)
        with self.assertRaises(ValueError):
            push_destination((0, 0), 'diagonal')

    def test_parser_preserves_legacy_and_rejects_invalid_push(self):
        legacy = {'object': 'object_1', 'target_zone': 'right_zone'}
        def parse(item):
            value = {'assessments': [], 'plan': [item], 'done': False}
            return parse_response(json.dumps(value))['plan'][0]
        self.assertEqual(parse(legacy), legacy)
        for direction in PUSH_DIRECTIONS:
            item = {'object': 'object_1', 'action': 'push', 'direction': direction}
            self.assertEqual(parse(item), item)
        for extra in ({}, {'direction': 'diagonal'}, {'direction': None}, {'direction': []},
                      {'direction': 'left', 'push_to': [0, 0]}):
            with self.assertRaises(ValueError):
                parse({'object': 'object_1', 'action': 'push', **extra})
        for item in ({'object': 'object_1', 'action': 'transport'},
                     {'object': 'object_1', 'action': 'stack', 'target_zone': 'right_zone'}):
            with self.assertRaises(ValueError):
                parse(item)

    def test_scripted_push_needs_no_pick_place(self):
        env = SortingEnv(layout='messy', seed=3, image_size=32)
        try:
            teacher = ScriptedSortingPolicy(env)
            teacher.set_subtask({'object': 'object_1', 'action': 'push',
                                 'push_to': push_destination(env.object_pos('object_1')[:2], 'toward_robot')})
            self.assertIsNone(teacher._pick_xy)
            self.assertIsNone(teacher._place_xy)
            self.assertTrue(np.isfinite(teacher.select_action()).all())
        finally:
            env.close()

    def test_runner_mixed_limits_feedback_and_A(self):
        # Rendering replaced before any environment is created: CPU physics only.
        def blank(env, camera='agentview', height=None, width=None):
            return np.zeros((height or env.image_size, width or env.image_size, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(SortingEnv, 'render', blank), \
                patch.object(runner, 'hardware', return_value={'diagnostic': 'CPU-only'}), \
                patch.object(runner, 'synchronize', return_value=None):
            root = Path(folder)
            scene = root / 'scene.json'
            dump(scene, freeze(template('open', seed=3, split='validation')))
            push = {'object': 'object_1', 'action': 'push', 'direction': 'toward_robot'}
            transport = {'object': 'object_1', 'target_zone': 'right_zone'}
            def trial(name, plan, condition='B'):
                policy = IdlePolicy()
                cfg = runner.TrialConfig(condition=condition, seconds=10, max_steps=8,
                    action_steps=1, video=False, device='cpu', image_size=32)
                if condition == 'C':
                    cfg.strategy = str(ROOT / 'final_project_v2/student_strategy_push_example.py')
                    cfg.max_calls = 2
                report = runner.run_trial(scene, root/name, cfg,
                    planner_adapter=Planner(plan), policy_adapter=policy)
                self.assertEqual(report['status'], 'completed', report.get('error'))
                return report, policy
            report, policy = trial('mixed', [push, push, transport, push])
            self.assertEqual(report['stop_reason'], 'push_attempt_limit')
            self.assertEqual(report['summary']['push_attempts'], {'object_1': 2})
            self.assertEqual(report['summary']['attempts'], {'object_1': 1})
            self.assertEqual(policy.tasks[:2], ['Push object 1 toward the robot.'] * 2)
            self.assertEqual(report['history'][0]['action'], push)
            self.assertEqual(report['history'][0]['direction'], 'toward_robot')
            self.assertEqual(report['history'][0]['action_type'], 'push')
            legacy, _ = trial('legacy', [transport] * 4)
            self.assertEqual(legacy['stop_reason'], 'object_attempt_limit')
            self.assertEqual(legacy['summary']['attempts'], {'object_1': 3})
            self.assertEqual(legacy['summary']['push_attempts'], {})
            a, policy = trial('A', [], 'A')
            self.assertEqual(a['summary']['planner_calls'], 0)
            expected = task_text(public_task(template('open')['command']))
            self.assertEqual(set(policy.tasks), {expected})
            c, _ = trial('C', [push, transport], 'C')
            self.assertEqual(c['summary']['planner_calls'], 2)
            self.assertEqual(c['summary']['push_attempts'], {'object_1': 2})
            self.assertEqual(c['summary']['attempts'], {'object_1': 1})
            replan = json.loads((root/'C'/'plan_002.json').read_text(encoding='utf-8'))
            self.assertEqual(replan['context']['execution_attempts']['push'], {'object_1': 1})
            # D stays transport-only by default. Inject a push choice solely to
            # verify its execution adapter resolves XY without mutating a plan.
            class Teacher:
                done = True
                tasks = []
                def __init__(self, env):
                    self.env = env
                def set_subtask(self, subtask):
                    expected = push_destination(self.env.object_pos('object_1')[:2], 'toward_robot')
                    np.testing.assert_array_equal(subtask['push_to'], expected)
                    self.tasks.append(subtask)
                def select_action(self):
                    return np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
            with patch.object(runner, 'classical_next', side_effect=[push, None]), \
                    patch.object(runner, 'ScriptedSortingPolicy', Teacher):
                cfg = runner.TrialConfig(condition='D', seconds=10, max_steps=8, video=False, device='cpu')
                d = runner.run_trial(scene, root/'D', cfg)
            self.assertEqual(d['status'], 'completed', d.get('error'))
            self.assertEqual(len(Teacher.tasks), 1)
            self.assertNotIn('push_to', push)


if __name__ == '__main__':
    unittest.main(verbosity=2)
