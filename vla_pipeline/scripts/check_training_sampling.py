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

# Identical physical rows can be divided into different whole/subtask episodes.
# Their sampled row stream must agree, including after checkpoint resume.
import torch
from vlapipe.training_sampling import FrameWeightedEpisodeSampler as FrameSampler
weights=torch.ones(100)
weights[[0,20,40,60,80]]=128
whole=FrameSampler([0,50],[50,100],frame_weights=weights,shuffle=True,seed=17)
parts=FrameSampler([0,20,40,60,80],[20,40,60,80,100],frame_weights=weights,shuffle=True,seed=17)
for epoch in [0,1]:
    whole.load_state_dict({'epoch':epoch,'start_index':0})
    parts.load_state_dict({'epoch':epoch,'start_index':0})
    expected=list(whole)
    assert list(parts)==expected
    parts.load_state_dict({'epoch':epoch,'start_index':19})
    assert list(parts)==expected[19:]
subset=FrameSampler([0,20],[20,40],episode_indices_to_use=[1],frame_weights=weights,
    shuffle=True,seed=17,absolute_to_relative_idx={n:n-20 for n in range(20,40)})
assert all(0<=value<20 for value in subset)
sequential=FrameSampler([0,50],[50,100],frame_weights=torch.zeros(1),shuffle=False)
assert list(sequential)==list(range(100))
for invalid in [torch.zeros(100),torch.full((100,),float('nan')),-torch.ones(100),torch.ones(50),torch.ones(100,1)]:
    try:
        FrameSampler([0,50],[50,100],frame_weights=invalid,shuffle=True)
        raise AssertionError('Invalid frame weights accepted')
    except ValueError:
        pass
print('PASS: equal physical-frame exposure across different episode views, deterministic resume, subset mapping, invalid weights rejected')
