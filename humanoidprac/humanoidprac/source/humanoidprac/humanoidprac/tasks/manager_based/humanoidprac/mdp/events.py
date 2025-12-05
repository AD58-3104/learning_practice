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

def EnvIdClassifier(num_of_envs: int):
    """
    環境のIDを分類するクラスを返す。
    この分類が同じもの同士は、環境としてまとめて扱う事が可能とする。例えば、この分類が同じ環境は同じエージェントのスコープで扱って良い
    """
    class _EnvIdClassifier:
        def __init__(self, num_of_envs: int):
            self.num_of_envs = num_of_envs
            # 上界を表すリスト(最後は省略されている)
            self.bound_list = torch.tensor([int(num_of_envs / 2)], device="cuda")
            self.num_of_classes = len(self.bound_list) + 1
            self.class_joint_id_list = [
                [1, 4, 8, 12],  # class 0 右
                [0, 3, 7, 11],  # class 1 左
            ]

        def get_class_env_bounds(self, class_id: int) -> tuple[int, int]:
            """
            指定されたクラスIDに対応する環境インデックスの範囲を取得する
            Args:
                class_id: クラスID
            Returns:
                (start, end) の環境インデックス範囲
            """
            start = 0 if class_id == 0 else int(self.bound_list[class_id - 1].item())
            end = int(self.bound_list[class_id].item()) if class_id < len(self.bound_list) else self.num_of_envs
            return (start, end)

        def classify_by_shape(self, all_env_tensor: torch.Tensor) -> list[torch.Tensor]:
            """
            環境IDを分類する
            Args:
                env_ids: 環境数分のテンソル
            Returns:
                分類されたテンソルのリスト
            """
            if all_env_tensor.shape[0] != self.num_of_envs:
                raise ValueError("EnvIdClassifier: classify_by_shape: このメソッドはテンソルを形で分類するので、全ての環境分のテンソルを渡す必要がある")
            return all_env_tensor.tensor_split(self.bound_list)
        
        def classify_by_envid(self, env_ids: torch.Tensor) -> list[torch.Tensor]:
            """
            渡された環境IDのテンソルを分類する
            Args:
                env_ids: 環境IDのテンソル
            Returns:
                分類されたテンソルのリスト
            """
            # boundaries[i-1] < input[m][n]...[l][x] <= boundaries[i]  これがFalseの時。つまり"以下"で分けられる。
            indices = torch.bucketize(env_ids, self.bound_list, right=False)
            counts = torch.bincount(indices, minlength=self.num_of_classes)

            return torch.split(env_ids, counts.tolist())
        
        def get_class_mask(self,class_id: int) -> torch.Tensor:
            """
            自分のクラスだけを有効にするマスクを取得する
            Args:
                class_id: クラスID
            Returns:
                マスクテンソル(num_envs,)
            """
            start, end = self.get_class_env_bounds(class_id)
            mask = torch.zeros(self.num_of_envs, dtype=torch.bool, device="cuda")
            mask[start:end] = True
            return mask
        
        def mask_other_classes(self, class_id: int, all_env_tensor: torch.Tensor) -> torch.Tensor:
            """
            自分のクラス以外をマスクする
            Args:
                class_id: クラスID
                all_env_tensor: 環境数分のテンソル
            Returns:
                マスクされたテンソル
            """
            if all_env_tensor.shape[0] != self.num_of_envs:
                raise ValueError("EnvIdClassifier: mask_other_classes: このメソッドはテンソルを形で分類するので、全ての環境分のテンソルを渡す必要がある")
            mask = self.get_class_mask(class_id)
            masked_tensor = torch.zeros_like(all_env_tensor)
            masked_tensor[mask] = all_env_tensor[mask]
            return masked_tensor

        def get_class_envs(self, class_id: int) -> int:
            """
            指定されたクラスIDに対応する環境数を取得する
            Args:
                class_id: クラスID
            Returns:
                環境数
            """
            # start, end = self.get_class_env_bounds(class_id)
            # return end - start
            return self.num_of_envs

    return _EnvIdClassifier(num_of_envs)


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

        # 全部300に戻してから指定した関節だけ変えるようにすれば多分良い
        # reset to 300
        self.joint_list_right = [
            "right_hip_yaw",
            "right_hip_roll",
            "right_hip_pitch",
            "right_knee",
        ]
        self.joint_list_left = [
            "left_hip_yaw",
            "left_hip_roll",
            "left_hip_pitch",
            "left_knee",
        ]

        self.right_legs = SceneEntityCfg()
        self.right_legs.name = "robot"
        self.right_legs.joint_names = self.joint_list_right
        self.right_legs.resolve(env.scene)

        self.left_legs = SceneEntityCfg()
        self.left_legs.name = "robot"
        self.left_legs.joint_names = self.joint_list_left
        self.left_legs.resolve(env.scene)

        # print(f"Right_legs joint_ids: {self.right_legs.joint_ids}")
        # print(f"Left_legs joint_ids: {self.left_legs.joint_ids}")
        # >>> Right_legs joint_ids: [1, 4, 8, 12]
        # >>> Left_legs joint_ids: [0, 3, 7, 11]

        torque_300 = [300.0 for _ in range(len(self.right_legs.joint_ids + self.left_legs.joint_ids))] # legsに入ってるのはtorso込みで9個のjoint
        self.torque_300_torch = torch.tensor(torque_300,device="cuda")
        self.log_file = None
        self.asset = env.scene["robot"]
        self.classifier = EnvIdClassifier(env.num_envs)

    def __del__(self):
        if self.log_file is not None:
            self.log_file.close()


    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        joint_torque: float,
        target_joint_cfg: SceneEntityCfg,
        normal_size: int = 0,
        logging: bool = False
    ):
        if env.num_envs != 1 and logging:
            raise ValueError("change_random_joint_torque(): Logging is only supported for single environment!!!")

        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cuda")
            print("change_random_joint_torque: env_ids is None, so set to all envs")
        else:
            env_ids = env_ids.cuda()

        # リセットする
        self.asset.write_joint_effort_limit_to_sim(self.torque_300_torch,(self.right_legs.joint_ids + self.left_legs.joint_ids), env_ids)

        import random

        # 環境IDを分類
        env_ids_right, env_ids_left = self.classifier.classify_by_envid(env_ids)

        # 右側をランダム選出
        target_joint_right = [joint_id for joint_id in target_joint_cfg.joint_ids if joint_id in self.right_legs.joint_ids]
        random_joint_right = random.choice(target_joint_right + [999 for _ in range(normal_size)])
        # 999が選ばれたら制限はしない
        if random_joint_right != 999:
            joint_torque_torch = torch.tensor(joint_torque,device="cuda")
            if len(env_ids_right) > 0:
                self.asset.write_joint_effort_limit_to_sim(joint_torque_torch, [random_joint_right], env_ids_right)

        # 左側をランダム選出
        target_joint_left = [joint_id for joint_id in target_joint_cfg.joint_ids if joint_id in self.left_legs.joint_ids]
        random_joint_left = random.choice(target_joint_left + [999 for _ in range(normal_size)])
        # 999が選ばれたら制限はしない
        if random_joint_left != 999:
            joint_torque_torch = torch.tensor(joint_torque,device="cuda")
            if len(env_ids_left) > 0:
                self.asset.write_joint_effort_limit_to_sim(joint_torque_torch, [random_joint_left], env_ids_left)

        if logging:
            if self.log_file is None:
                self.log_file = open("joint_torque_event_log.dat","w")
                self.log_file.write("common_step_counter,target_joint_id\n")
            if random_joint_left != 999 or random_joint_right != 999: # 999はログに残さない。制限掛かってないので。
                self.log_file.write(f"{env.common_step_counter},{random_joint_left},{random_joint_right}\n")