import torch
from matplotlib import pyplot as plt

from torch_robotics.environment.env_base import EnvBase
from torch_robotics.environment.primitives import ObjectField, MultiSphereField
from torch_robotics.environment.utils import create_grid_spheres
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvGridCircles2D(EnvBase):

    def __init__(self, tensor_args=None, **kwargs):
        obj_list = create_grid_spheres(rows=7, cols=7, heights=0, radius=0.1, tensor_args=tensor_args)

        super().__init__(
            name=self.__class__.__name__,
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),  # environment limits
            obj_fixed_list=obj_list,
            tensor_args=tensor_args,
            **kwargs
        )

    def get_rrt_connect_params(self):
        params = dict(
            n_iters=10000,
            step_size=0.01,
            n_radius=0.3,
            n_pre_samples=50000,
            max_time=15
        )
        return params

    def get_gpmp_params(self):
        params = dict(
            opt_iters=100,
            num_samples=64,
            sigma_start=1e-5,
            sigma_gp=1e-2,
            sigma_goal_prior=1e-5,
            sigma_coll=1e-5,
            step_size=5e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.2,
            sigma_start_sample=1e-4,
            sigma_goal_sample=1e-4,
            sigma_gp_sample=0.2,
            solver_params={
                'delta': 1e-2,
                'trust_region': True,
                'method': 'cholesky',
            },
        )
        return params

    def get_sgpmp_params(self):
        params = dict(
            opt_iters=100,
            num_samples=64,
            sigma_start=1e-5,
            sigma_gp=1e-2,
            sigma_goal_prior=1e-5,
            sigma_coll=1e-5,
            step_size=5e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.2,
            sigma_start_sample=1e-4,
            sigma_goal_sample=1e-4,
            sigma_gp_sample=0.2,
            temperature=1.,
        )
        return params
    
    def get_mpot_params(self):
        solver_params = dict(
            reg=0.01,  # entropic regularization lambda
            num_probe=5,
            numInnerItermax=5,
            stopThr=8e-2,
            innerStopThr=1e-5,
            verbose=False,
        )

        params = dict(
            opt_iters=100,
            solver_params=solver_params,
            step_radius=0.38,
            probe_radius=0.5,  # probe radius > step radius
            polytope='cube',  # 'random' | 'simplex' | 'orthoplex' | 'cube'; 'random' option is added for ablations, not recommended for general use
            eps_annealing=0.02,
            num_bpoint=50,  # number of random points on the $4$-sphere, when polytope == 'random' (i.e. no polytope structure)
            num_probe=5,  # number of probes points for each polytope vertices
            pos_limits=[-1, 1],
            vel_limits=[-1, 1],
            w_smooth=1e-7,
            w_coll=2.4e-3,
            sigma_gp=0.13,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=1.6,
        )
        return params


if __name__ == '__main__':
    env = EnvGridCircles2D(precompute_sdf_obj_fixed=True, tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()