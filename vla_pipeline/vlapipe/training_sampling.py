"""Optional train-only curriculum for rare approach frames; no label changes."""
import torch
from lerobot.datasets.sampler import EpisodeAwareSampler


class StartWeightedEpisodeSampler(EpisodeAwareSampler):
    def __init__(self,*args,start_frame_weight=1.0,start_frames=8,**kwargs):
        super().__init__(*args,**kwargs)
        if start_frame_weight<1 or start_frames<1:
            raise ValueError('Start weight must be >=1 and start_frames positive')
        self.start_frame_weight=start_frame_weight
        self._sampling_weights=torch.ones(self._num_frames,dtype=torch.float64)
        begin=0
        for end in self._cum_lengths:
            self._sampling_weights[begin:min(begin+start_frames,int(end))]=start_frame_weight
            begin=int(end)

    def _iter_epoch(self,epoch,start):
        if not self.shuffle or self.start_frame_weight==1:
            yield from super()._iter_epoch(epoch,start)
            return
        order=torch.multinomial(self._sampling_weights,self._num_frames,replacement=True,
                                generator=self._epoch_generator(epoch))
        for position in order[start:]:
            yield self._frame_index(int(position))
