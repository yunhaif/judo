# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np
import numpy.typing as npt
import torch
from mujoco import (
    MjModel,
    mj_differentiatePos,
    mj_fwdPosition,
    mj_fwdVelocity,
    mj_integratePos,
    mj_sensorPos,
    mj_sensorVel,
    mj_step1,
    rollout,
)
from torch import FloatTensor, IntTensor, tensor



def get_joint_dimensions(
    joint_ids: npt.ArrayLike, state_address: npt.ArrayLike, state_length: int
) -> IntTensor:
    """Given a list of joint ids, and the list of addresses in the states for the joints.
    We return the dimensions in the state corresponding to the list of joint ids."""
    dims = []
    for idx in joint_ids:
        start = state_address[idx]  # start index in state
        if idx + 1 < len(state_address):
            end = state_address[idx + 1]  # end index in state
        else:
            end = state_length
        dims.extend(list(range(start, end)))
    return tensor(dims)


def decompose_state_dimensions(
    model: MjModel,
) -> Tuple[IntTensor, IntTensor, IntTensor, IntTensor]:
    """Decompose the states indices in 4 groups, split between positions or velocities and
    actuated or not actuated."""
    nq = model.nq
    nv = model.nv

    joints = model.actuator_trnid[:, 0]
    pos_address = model.jnt_qposadr
    vel_address = model.jnt_dofadr

    actuated_pos = get_joint_dimensions(joints, pos_address, nq)
    actuated_vel = nq + get_joint_dimensions(joints, vel_address, nv)

    unactuated_pos = tensor([i for i in range(nq) if i not in actuated_pos])
    unactuated_vel = tensor([i for i in range(nq, nq + nv) if i not in actuated_vel])

    return actuated_pos, actuated_vel, unactuated_pos, unactuated_vel

