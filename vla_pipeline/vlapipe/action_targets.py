"""Versioned neural action targets; external robot actions remain delta7."""
import numpy as np
from . import config as C


def validate_descriptor(representation):
    if representation.get('version')!=1 or representation.get('external_action')!='delta7':
        raise ValueError('Unsupported action target metadata version/interface')
    if representation.get('name') not in ('delta','absolute_position'):
        raise ValueError('Unsupported neural action representation')
    if representation['name']=='absolute_position':
        low=np.asarray(representation.get('position_low'),dtype=float)
        high=np.asarray(representation.get('position_high'),dtype=float)
        if low.shape!=(3,) or high.shape!=(3,) or not np.isfinite([low,high]).all() or not np.all(high>low):
            raise ValueError('Invalid XYZ normalization bounds in model metadata')
    return representation


def descriptor(name='delta'):
    if name=='delta':
        return {'name':'delta','version':1,'external_action':'delta7'}
    if name!='absolute_position':
        raise ValueError('Unknown neural action representation: '+name)
    from .controller import OSCController
    return {'name':name,'version':1,'external_action':'delta7',
        'position_low':OSCController.BOUNDS_LO.tolist(),'position_high':OSCController.BOUNDS_HI.tolist(),
        'note':'Head xyz encodes the next commanded end-effector position; rotation remains delta, gripper absolute. No object state is used.'}


def encode_episode_actions(ep,representation):
    actions=ep.actions.copy()
    if representation['name']=='absolute_position':
        low=np.asarray(representation['position_low'])
        high=np.asarray(representation['position_high'])
        actions[:,:3]=np.clip(2*(ep.target_pos[1:len(ep)+1]-low)/(high-low)-1,-1,1)
    return actions


def to_environment_action(head,representation,controller_target=None):
    action=np.asarray(head,dtype=np.float32).copy()
    if action.shape!=(C.ACTION_DIM,) or not np.isfinite(action).all():
        raise ValueError('Expected finite action7 from the neural head')
    if representation['name']=='absolute_position':
        if controller_target is None:
            raise ValueError('This model needs set_controller_target(env.ctrl.target_pos) before each action; the target is robot actuator state, not an object position.')
        low=np.asarray(representation['position_low'])
        high=np.asarray(representation['position_high'])
        position=low+(np.clip(action[:3],-1,1)+1)*(high-low)/2
        action[:3]=(position-np.asarray(controller_target))/C.MAX_DPOS
    elif representation['name']!='delta':
        raise ValueError('Unsupported action representation')
    action=np.clip(action,-1,1)
    action[6]=C.GRIPPER_CLOSE_CMD if action[6]>=0 else C.GRIPPER_OPEN_CMD
    return action.astype(np.float32)
