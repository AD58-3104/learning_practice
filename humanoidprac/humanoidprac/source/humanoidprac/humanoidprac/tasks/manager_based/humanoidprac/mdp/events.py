from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import omni.physics.tensors.impl.api as physx

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class change_joint_torque(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            ValueError: If the asset is not a RigidObject or an Articulation.
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset, (RigidObject, Articulation)):
            raise ValueError(
                f"Joint setting change term 'change_joint_torque' not supported for asset: '{self.asset_cfg.name}'"
                f" with type: '{type(self.asset)}'."
            )

        # obtain number of shapes per body (needed for indexing the material properties correctly)
        # note: this is a workaround since the Articulation does not provide a direct way to obtain the number of shapes
        #  per body. We use the physics simulation view to obtain the number of shapes per body.
        if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(None):
            self.num_shapes_per_body = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)
            # ensure the parsing is correct
            num_shapes = sum(self.num_shapes_per_body)
            expected_shapes = self.asset.root_physx_view.max_shapes
            if num_shapes != expected_shapes:
                raise ValueError(
                    "Joint setting change term 'change_joint_torque' failed to parse the number of shapes per body."
                    f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
                )
        else:
            # in this case, we don't need to do special indexing
            self.num_shapes_per_body = None


    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        joint_torque: Sequence[float],
        asset_cfg: SceneEntityCfg,
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cuda")
        else:
            env_ids = env_ids.cuda()
        joint_torque_torch = torch.tensor(joint_torque,device="cuda")
        # apply to simulation
        self.asset.write_joint_effort_limit_to_sim(joint_torque_torch,asset_cfg.joint_ids, env_ids)


class change_random_joint_torque(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            ValueError: If the asset is not a RigidObject or an Articulation.
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset, (RigidObject, Articulation)):
            raise ValueError(
                f"Joint setting change term 'change_joint_torque' not supported for asset: '{self.asset_cfg.name}'"
                f" with type: '{type(self.asset)}'."
            )

        # obtain number of shapes per body (needed for indexing the material properties correctly)
        # note: this is a workaround since the Articulation does not provide a direct way to obtain the number of shapes
        #  per body. We use the physics simulation view to obtain the number of shapes per body.
        if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(None):
            self.num_shapes_per_body = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)
            # ensure the parsing is correct
            num_shapes = sum(self.num_shapes_per_body)
            expected_shapes = self.asset.root_physx_view.max_shapes
            if num_shapes != expected_shapes:
                raise ValueError(
                    "Joint setting change term 'change_joint_torque' failed to parse the number of shapes per body."
                    f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
                )
        else:
            # in this case, we don't need to do special indexing
            self.num_shapes_per_body = None

        # 全部300に戻してから指定した関節だけ変えるようにすれば多分良い
        # reset to 300
        self.joint_list = [
            "right_hip_yaw",
            "left_hip_yaw",
            "right_hip_roll",
            "left_hip_roll",
            "right_hip_pitch",
            "left_hip_pitch",
            "right_knee",
            "left_knee",
        ]
        self.legs = SceneEntityCfg()
        self.legs.name = "robot"
        self.legs.joint_names = self.joint_list
        self.legs.resolve(env.scene)
        torque_300 = [300.0 for _ in range(len(self.joint_list))] # legsに入ってるのはtorso込みで9個のjoint
        self.torque_300_torch = torch.tensor(torque_300,device="cuda")

        knees = SceneEntityCfg()
        knees.name = "robot"
        knees.joint_names = ["right_knee","left_knee"]
        knees.resolve(env.scene)

        # self.random_joint_ids = self.legs.joint_ids + knees.joint_ids + knees.joint_ids # 膝が選ばれる確率は他の3倍
        self.random_joint_ids = knees.joint_ids


    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        joint_torque: float,
        asset_cfg: SceneEntityCfg,
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cuda")
        else:
            env_ids = env_ids.cuda()
        # リセットする
        self.asset.write_joint_effort_limit_to_sim(self.torque_300_torch,self.legs.joint_ids, env_ids)

        # ランダムに選んだ関節のトルクを制限する
        import random
        random_joint = random.choice(self.random_joint_ids)
        joint_torque_torch = torch.tensor(joint_torque,device="cuda")
        self.asset.write_joint_effort_limit_to_sim(joint_torque_torch, [random_joint], env_ids)