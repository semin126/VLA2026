"""Verify train-only weighting, subset mapping, and deterministic epoch resume."""
import _bootstrap  # noqa: F401
from vlapipe.training_sampling import StartWeightedEpisodeSampler as Sampler

sampler=Sampler([0,20],[20,40],episode_indices_to_use=[1],shuffle=True,seed=3,
    start_frame_weight=10,start_frames=3)
values=list(sampler)
assert all(20<=value<40 for value in values)
sampler.load_state_dict({'epoch':0,'start_index':7})
assert list(sampler)==values[7:]
mapped=Sampler([0,20],[20,40],episode_indices_to_use=[1],shuffle=True,seed=3,
    absolute_to_relative_idx={n:n-20 for n in range(20,40)},start_frame_weight=10,start_frames=3)
assert list(mapped)==[n-20 for n in values]
plain=Sampler([0],[40],shuffle=True,seed=3)
assert sorted(plain)==list(range(40))
evaluation=Sampler([0],[40],shuffle=False,start_frame_weight=10)
assert list(evaluation)==list(range(40))
weighted=Sampler([0],[5000],shuffle=True,seed=3,start_frame_weight=10,start_frames=500)
assert 2400<sum(index<500 for index in weighted)<2900
print('PASS: selected episodes and relative mapping, deterministic resume, weighted approaches, native default/evaluation')
