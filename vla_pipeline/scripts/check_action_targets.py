"""The pose head adapter preserves external delta7 and cannot accumulate a fixed pose bias."""
import _bootstrap  # noqa: F401
import numpy as np
from vlapipe.action_targets import descriptor,to_environment_action
from vlapipe import config as C

pose=descriptor('absolute_position')
low,high=np.asarray(pose['position_low']),np.asarray(pose['position_high'])
desired=(low+high)/2
head=np.zeros(7,dtype=np.float32)
head[3:]=[.1,-.2,.3,-1]
target=desired+np.array([.1,-.1,.05])
for _ in range(20):
    action=to_environment_action(head,pose,target)
    assert action.shape==(7,) and np.all(np.abs(action)<=1)
    assert np.allclose(action[3:],head[3:])
    target=np.clip(target+action[:3]*C.MAX_DPOS,low,high)
assert np.allclose(target,desired,atol=1e-7)
assert np.allclose(to_environment_action(head,pose,target)[:3],0,atol=1e-6)
try:
    to_environment_action(head,pose)
    raise AssertionError('Missing robot command state was silently guessed')
except ValueError:
    pass
legacy=to_environment_action(head,descriptor())
assert np.array_equal(legacy,head)
print('PASS: external delta7, fixed pose converges without accumulation, missing robot command rejected, legacy unchanged')
