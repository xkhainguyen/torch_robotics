import torch
import time

from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor, link_quat_from_link_tensor
from torch_robotics.torch_kinematics_tree.models.robots import DifferentiableCartpole2L, DifferentiablePendulum1L, DifferentiablePendulum2L, Differentiable2LinkPlanar
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA


if __name__ == "__main__":
    seed = 1
    fix_random_seed(seed)

    batch_size = 1
    device = "cpu"
    # device = "cuda:0"

    print("\n===========================Pendulum Model===============================")
    robot = DifferentiablePendulum2L(device=device)
    robot.print_link_names()
    print(robot.print_dynamics_info())
    print(robot._n_dofs)
    for i in range(10):
        q = torch.rand(batch_size, robot._n_dofs, requires_grad=True).to(device)
        qd = torch.rand(batch_size, robot._n_dofs, requires_grad=True).to(device)
        qdd = torch.rand(batch_size, robot._n_dofs, requires_grad=True).to(device)
        # f = robot.compute_inverse_dynamics(q, qd, qdd)
        f = torch.rand(batch_size, robot._n_dofs, requires_grad=True).to(device)
        qdd = robot.compute_forward_dynamics(q, qd, f)
        grad = torch.autograd.grad(qdd, (q, qd, f))

    with TimerCUDA() as t:
        for i in range(1000):
            q = torch.rand(batch_size, robot._n_dofs, requires_grad=True).to(device)
            qd = torch.rand(batch_size, robot._n_dofs, requires_grad=True).to(device)
            qdd = torch.rand(batch_size, robot._n_dofs, requires_grad=True).to(device)
            # f = robot.compute_inverse_dynamics(q, qd, qdd)
            f = torch.rand(batch_size, robot._n_dofs, requires_grad=True).to(device)
            qdd = robot.compute_forward_dynamics(q, qd, f)
            # print(qdd)
            grad = torch.autograd.grad(qdd, (q, qd, f))
            # print(grad)
            # print('Error:', torch.norm(qdd - qdd2))

    print(f"Computational Time {t.elapsed:.4f}")
