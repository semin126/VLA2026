"""CPU-only grasp-axis contract checks. No model loading or rendering."""
import _bootstrap  # noqa: F401
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from vlapipe import numbered
from vlapipe.controller import yaw_to_grasp_quat
from vlapipe.planner import axis_instruction, instruction_text, grasp_axis_yaw
from vlapipe.policy import ScriptedSortingPolicy
from vlapipe.final_project_v2.common import parse_response


class AxisChecks(unittest.TestCase):
    def setUp(self):
        numbered.activate(count=10)

    def test_instruction_exact_and_legacy(self):
        for n in (1, 6, 10):
            for zone in ('left_zone', 'right_zone'):
                self.assertEqual(axis_instruction(f'object_{n}', zone), instruction_text(f'object_{n}', zone)[0])
                label = zone.replace('_', ' ').replace('zone', 'storage zone')
                for axis in ('left-right', 'front-back'):
                    self.assertEqual(axis_instruction(f'object_{n}', zone, axis),
                        f'Pick up the cube number {n} with the fingers {axis} and place it in the {label}.')

    def test_face_axes_not_world_snap(self):
        self.assertEqual(grasp_axis_yaw(0, 'left-right'), 0)
        self.assertEqual(grasp_axis_yaw(0, 'front-back'), math.pi / 2)
        yaw = math.radians(20)
        self.assertEqual(grasp_axis_yaw(yaw, 'left-right'), yaw)
        self.assertEqual(grasp_axis_yaw(yaw, 'front-back'), yaw + math.pi / 2)
        self.assertAlmostEqual(grasp_axis_yaw(math.pi / 2, 'left-right'), 2 * math.pi)

    def test_teacher_refresh_and_legacy(self):
        env = SimpleNamespace(object_pos=Mock(return_value=np.array([.1, .3, .8])),
            object_yaw=Mock(return_value=0.), grasp_pose=Mock(return_value=(None, np.array([1., 2., 3., 4.]))))
        teacher = ScriptedSortingPolicy(env)
        task = {'object': 'object_1', 'target_zone': 'right_zone'}
        teacher.set_subtask(task)
        np.testing.assert_array_equal(teacher._grasp_quat, [1., 2., 3., 4.])
        teacher._refresh_pick()
        np.testing.assert_array_equal(teacher._grasp_quat, [1., 2., 3., 4.])
        teacher.set_subtask({**task, 'grasp_axis': 'front-back'})
        np.testing.assert_array_equal(teacher._grasp_quat, yaw_to_grasp_quat(math.pi / 2))
        env.object_yaw.return_value = math.radians(20)
        env.object_pos.return_value = np.array([.2, .4, .8])
        teacher._refresh_pick()
        np.testing.assert_array_equal(teacher._grasp_quat, yaw_to_grasp_quat(math.radians(110)))
        self.assertEqual(teacher._pick_xy, (.2, .4))

    def test_schema_accepts_axes_and_rejects_invalid(self):
        item = {'object': 'object_1', 'target_zone': 'right_zone', 'grasp_axis': 'front-back'}
        assessment = {'object': 'object_1', 'accessible': True, 'blockers': [], 'reason': 'image',
                      'free_axes': ['front-back']}
        def parse(plan, assess=assessment):
            return parse_response(json.dumps({'assessments': [assess], 'plan': [plan], 'done': False}))
        self.assertEqual(parse(item)['plan'][0], item)
        for value in (None, [], 'diagonal', 1):
            with self.assertRaises(ValueError):
                parse({**item, 'grasp_axis': value})
        for value in (None, 'front-back', ['diagonal'], [[]], ['left-right', 'left-right']):
            with self.assertRaises(ValueError):
                parse(item, {**assessment, 'free_axes': value})
        for value in ([], ['left-right'], ['front-back'], ['left-right', 'front-back']):
            self.assertEqual(parse(item, {**assessment, 'free_axes': value})['assessments'][0]['free_axes'], value)
        with self.assertRaises(ValueError):
            parse({'object': 'object_1', 'action': 'push', 'direction': 'left', 'grasp_axis': 'front-back'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
