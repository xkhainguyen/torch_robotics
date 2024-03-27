import torch
import time

from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor, link_quat_from_link_tensor
from torch_robotics.torch_kinematics_tree.models.robots import DifferentiableCartpole2L
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA


if __name__ == "__main__":
    seed = 1
    fix_random_seed(seed)

    batch_size = 1
    # device = "cpu"
    device = "cuda:0"

    print("\n===========================Cartpole Model===============================")
    diff_cp2l = DifferentiableCartpole2L(device=device)
    diff_cp2l.print_link_names()
    print(diff_cp2l.print_dynamics_info())
    print(diff_cp2l._n_dofs)
    with TimerCUDA() as t:
        q = torch.rand(batch_size, diff_cp2l._n_dofs).to(device) * 2 - 1
        qd = torch.rand(batch_size, diff_cp2l._n_dofs).to(device)
        qdd = torch.rand(batch_size, diff_cp2l._n_dofs).to(device)
        f = diff_cp2l.compute_inverse_dynamics(q, qd, qdd)
        qdd2 = diff_cp2l.compute_forward_dynamics(q, qd, f)
        print('Error:', torch.norm(qdd - qdd2))

    print(f"Computational Time {t.elapsed:.4f}")
