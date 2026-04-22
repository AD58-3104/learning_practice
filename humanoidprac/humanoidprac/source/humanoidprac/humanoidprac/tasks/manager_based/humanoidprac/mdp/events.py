from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal, Sequence

import omni.physics.tensors.impl.api as physx

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DelayedFaultNotifier:
    """
    故障の遅延通知を管理するクラス。

    このクラスは以下の機能を提供します：
    1. 各エピソード開始時に遅延ステップ数を一様分布からサンプリング
    2. 故障発生時に故障関節をバッファに記録
    3. 指定したステップ数経過後に故障を通知

    使用方法:
        notifier = DelayedFaultNotifier(num_envs, delay_range=(5, 20))

        # リセット時に遅延をサンプリング
        notifier.on_episode_reset(env_ids)

        # 故障発生時にバッファに記録
        notifier.register_fault(env_ids, joint_ids)

        # 毎ステップ呼び出して通知をチェック
        notified_envs, notified_joints = notifier.step()
    """

    def __init__(self, num_envs: int, delay_range: tuple[int, int] = (5, 20), device: str = "cuda"):
        """
        Args:
            num_envs: 環境の数
            delay_range: 遅延ステップ数の範囲 (min, max)。一様分布からサンプリングされる
            device: テンソルを配置するデバイス
        """
        self.num_envs = num_envs
        self.delay_min = int(delay_range[0])
        self.delay_max = int(delay_range[1])
        self.device = device

        # 各環境の遅延ステップ数 (エピソード開始時にサンプリング)
        self.sampled_delay = torch.zeros(num_envs, dtype=torch.long, device=device)

        # 各環境で現在故障中の関節ID (-1は故障なし)
        self.current_fault_joint = torch.full((num_envs,), -1, dtype=torch.long, device=device)

        # 故障発生からのカウンタ (-1は故障発生していない)
        self.fault_counter = torch.full((num_envs,), -1, dtype=torch.long, device=device)

        # 既に通知済みかどうか
        self.notified = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def on_episode_reset(self, env_ids: torch.Tensor | None = None):
        """
        エピソードのリセット時に呼び出す。遅延ステップ数をサンプリングし、バッファをリセットする。

        Args:
            env_ids: リセットする環境のID。Noneの場合は全環境
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # 一様分布から自然数をサンプリング
        num_reset = len(env_ids)
        self.sampled_delay[env_ids] = torch.randint(
            self.delay_min, self.delay_max + 1, (num_reset,), device=self.device
        )

        # バッファをリセット
        self.current_fault_joint[env_ids] = -1
        self.fault_counter[env_ids] = -1
        self.notified[env_ids] = False

    def register_fault(self, env_ids: torch.Tensor, joint_ids: torch.Tensor | int):
        """
        故障を登録する。change_random_joint_torqueイベントから呼び出される。

        Args:
            env_ids: 故障が発生した環境のID
            joint_ids: 故障した関節のID。スカラーまたは環境ごとに異なる場合はテンソル
        """
        if len(env_ids) == 0:
            return

        # joint_idsがスカラーの場合はテンソルに変換
        if isinstance(joint_ids, int):
            joint_ids = torch.full((len(env_ids),), joint_ids, dtype=torch.long, device=self.device)
        elif not isinstance(joint_ids, torch.Tensor):
            joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)

        # 故障関節を記録
        self.current_fault_joint[env_ids] = joint_ids.to(self.device)

        # カウンタを0から開始
        self.fault_counter[env_ids] = 0

        # 通知フラグをリセット
        self.notified[env_ids] = False

    def step(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        毎ステップ呼び出す。カウンタをインクリメントし、遅延に達した環境の通知を返す。

        Returns:
            (notified_env_ids, notified_joint_ids):
                今回通知すべき環境IDと対応する故障関節ID
        """
        # 故障が発生している環境（カウンタが0以上）のカウンタをインクリメント
        active_faults = self.fault_counter >= 0
        self.fault_counter[active_faults] += 1

        # 遅延に達し、まだ通知していない環境を特定
        should_notify = (
            (self.fault_counter >= self.sampled_delay) &
            (~self.notified) &
            (self.current_fault_joint >= 0)
        )

        notified_env_ids = torch.nonzero(should_notify, as_tuple=False).squeeze(-1)
        notified_joint_ids = self.current_fault_joint[notified_env_ids]

        # 通知済みフラグを立てる
        self.notified[notified_env_ids] = True

        return notified_env_ids, notified_joint_ids

    def get_current_faults(self) -> torch.Tensor:
        """
        各環境で現在故障中の関節IDを取得する。

        Returns:
            (num_envs,) の関節IDテンソル。故障なしの環境は-1
        """
        return self.current_fault_joint.clone()

    def get_notified_faults(self) -> torch.Tensor:
        """
        各環境で通知済みの故障関節IDを取得する。
        通知されていない場合は-1を返す。

        Returns:
            (num_envs,) の関節IDテンソル。通知されていない環境は-1
        """
        result = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        result[self.notified] = self.current_fault_joint[self.notified]
        return result

    def get_delay_remaining(self) -> torch.Tensor:
        """
        各環境で通知までの残りステップ数を取得する。

        Returns:
            (num_envs,) の残りステップ数テンソル。故障なしまたは通知済みは-1
        """
        result = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        active = (self.fault_counter >= 0) & (~self.notified)
        result[active] = self.sampled_delay[active] - self.fault_counter[active]
        return result


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
            self.class_joint_id_list = [
                # [1,4,8,12],  # class 0 右
                # [0,3,7,11],  # class 1 左
                [1],  # class 0 右
                [4],
                [8],
                [12],
                [0], 
                [3],
                [7],
                [11],  # class 1 左
            ]
            self.num_of_classes = len(self.class_joint_id_list)
            bound_list = []
            for i in range(len(self.class_joint_id_list)-1):
                # joint_idの次の環境IDを境界とする
                bound_list.append((self.num_of_envs // self.num_of_classes) * (i + 1))
            self.bound_list = torch.tensor(bound_list, device="cuda")
            print(f"EnvIdClassifier: boundaries: {self.bound_list.tolist()}")
            assert len(self.bound_list) + 1 == self.num_of_classes, "EnvIdClassifier: クラス数と境界数が合わない"


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
            start, end = self.get_class_env_bounds(class_id)
            return end - start

    return _EnvIdClassifier(num_of_envs)

class reset_all_joint_torques(ManagerTermBase):
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
        self.asset = env.scene["robot"]

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cuda")
        else:
            env_ids = env_ids.cuda()
        # リセットする
        self.asset.write_joint_effort_limit_to_sim(self.torque_300_torch,(self.right_legs.joint_ids + self.left_legs.joint_ids), env_ids)

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
        self.joint_torque_right = {
            1: 2.0,
            4: 10.0,
            8: 20.0,
            12: 50.0,
        }

        self.joint_list_left = [
            "left_hip_yaw",
            "left_hip_roll",
            "left_hip_pitch",
            "left_knee",
        ]
        self.joint_torque_left = {
            0: 2.0,
            3: 10.0,
            7: 20.0,
            11: 50.0,
        }

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
        self.log_files = [None for _ in range(env.num_envs)]
        self.asset = env.scene["robot"]
        self.classifier = EnvIdClassifier(env.num_envs)

    def __del__(self):
        for f in self.log_files:
            if f is not None:
                f.close()


    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        joint_torque: float,    # これは単一環境でのみ利用される
        target_joint_cfg: SceneEntityCfg,
        normal_size: int = 0,
        logging: bool = False
    ):
        # if env.num_envs != 1 and logging:
        #     raise ValueError("change_random_joint_torque(): Logging is only supported for single environment!!!")

        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cuda")
            print("change_random_joint_torque: env_ids is None, so set to all envs")
        else:
            env_ids = env_ids.cuda()

        # # リセットする
        # self.asset.write_joint_effort_limit_to_sim(self.torque_300_torch,(self.right_legs.joint_ids + self.left_legs.joint_ids), env_ids)

        import random

        robot = env.unwrapped.scene["robot"]
        joint_effort_limits = robot.data.joint_effort_limits
        reseted_limits = joint_effort_limits < (joint_torque + 1)
        reseted_limits = reseted_limits.sum(dim=1) > 0  # 環境ごとに一つでも制限が変わっていたらTrue
        not_reseted_limits_env_indices = torch.nonzero(~reseted_limits, as_tuple=False).squeeze(-1)
        # print(f"not reseted limits env ids: {not_reseted_limits_indices.tolist()}")
        


        # 環境が1以上あり複数のクラスに分離される場合
        if env.num_envs > 1:
            # 環境IDを8つのクラスに分類
            classified_env_ids = self.classifier.classify_by_envid(not_reseted_limits_env_indices)
            # クラス0-3: 右脚（joint 1, 4, 8, 12）
            # クラス4-7: 左脚（joint 0, 3, 7, 11）
            # 右脚と左脚のグループに統合
            right_tensors = [classified_env_ids[i] for i in range(4) if len(classified_env_ids[i]) > 0]
            # env_ids_right = classified_env_ids[0]
            # env_ids_left = classified_env_ids[1]
            left_tensors = [classified_env_ids[i] for i in range(4, 8) if len(classified_env_ids[i]) > 0]
            env_ids_right = torch.cat(right_tensors) if len(right_tensors) > 0 else torch.tensor([], dtype=torch.long, device="cuda")
            env_ids_left = torch.cat(left_tensors) if len(left_tensors) > 0 else torch.tensor([], dtype=torch.long, device="cuda")

            # 右側をランダム選出
            target_joint_right = [joint_id for joint_id in target_joint_cfg.joint_ids if joint_id in self.right_legs.joint_ids]
            random_joint_right = random.choice(target_joint_right + [999 for _ in range(normal_size)])
            # 999が選ばれたら制限はしない
            if random_joint_right != 999:
                joint_torque_torch = torch.tensor(self.joint_torque_right[random_joint_right],device="cuda")
                if len(env_ids_right) > 0:
                    self.asset.write_joint_effort_limit_to_sim(joint_torque_torch, [random_joint_right], env_ids_right)
                    # print(f"change_random_joint_torque: right joint {random_joint_right} torque changed to {joint_torque} for envs {env_ids_right.tolist()}")

            # 左側をランダム選出
            target_joint_left = [joint_id for joint_id in target_joint_cfg.joint_ids if joint_id in self.left_legs.joint_ids]
            random_joint_left = random.choice(target_joint_left + [999 for _ in range(normal_size)])
            # 999が選ばれたら制限はしない
            if random_joint_left != 999:
                joint_torque_torch = torch.tensor(self.joint_torque_left[random_joint_left],device="cuda")
                if len(env_ids_left) > 0:
                    self.asset.write_joint_effort_limit_to_sim(joint_torque_torch, [random_joint_left], env_ids_left)
                    # print(f"change_random_joint_torque: left joint {random_joint_right} torque changed to {joint_torque} for envs {env_ids_left.tolist()}")
        else:
            # 単一環境の場合
            target_joint_ids = target_joint_cfg.joint_ids
            single_random_joint = random.choice(target_joint_ids + [999 for _ in range(normal_size)])
            # 999が選ばれたら制限はしない
            if single_random_joint != 999:
                if single_random_joint in self.joint_torque_right:
                    joint_torque = self.joint_torque_right[single_random_joint]
                elif single_random_joint in self.joint_torque_left:
                    joint_torque = self.joint_torque_left[single_random_joint]
                joint_torque_torch = torch.tensor(joint_torque,device="cuda")
                self.asset.write_joint_effort_limit_to_sim(joint_torque_torch, [single_random_joint], env_ids)


        if logging:
            # if env.num_envs != 1:
            #     raise ValueError("change_random_joint_torque(): Logging is only supported for single environment!!!")
            # if env_ids is None:
            #     env_ids = torch.arange(env.scene.num_envs, device="cuda")
            # 左側の関節IDのログ
            for idx in env_ids_left.tolist():
                if self.log_files[idx] is None:
                    self.log_files[idx] = open(f"./nn_data/joint_torque_event_log_env_{idx}.dat","w")
                    self.log_files[idx].write("common_step_counter,target_joint_id\n")
                if random_joint_left != 999: # 999はログに残さない。制限掛かってないので。
                    self.log_files[idx].write(f"{env.common_step_counter},{random_joint_left}\n")
            # 右側の関節IDのログ
            for idx in env_ids_right.tolist():
                if self.log_files[idx] is None:
                    self.log_files[idx] = open(f"./nn_data/joint_torque_event_log_env_{idx}.dat","w")
                    self.log_files[idx].write("common_step_counter,target_joint_id\n")
                if random_joint_right != 999: # 999はログに残さない。制限掛かってないので。
                    self.log_files[idx].write(f"{env.common_step_counter},{random_joint_right}\n")


class change_random_joint_torque_with_delayed_notification(ManagerTermBase):
    """
    故障を発生させ、遅延通知機能を持つイベントクラス。

    このクラスは以下の機能を提供します：
    1. ランダムな関節の故障（トルク制限）を発生させる
    2. 故障関節をバッファ（current_fault_buffer）に記録する
    3. DelayedFaultNotifierを使用して遅延通知を管理する

    使用方法:
        EventCfgにて以下のように設定:

        change_random_joint_torque_with_delayed_notification = EventTerm(
            func=mdp.change_random_joint_torque_with_delayed_notification,
            mode="interval",
            interval_range_s=(5.0, 10.0),
            params={
                "target_joint_cfg": SceneEntityCfg("robot", joint_names=["right_knee", "left_knee"]),
                "joint_torque": 50.0,
                "delay_range": (5, 20),  # 通知遅延のステップ数範囲
            }
        )

    バッファへのアクセス:
        # 現在の故障関節を取得
        fault_buffer = env.scene["robot"]._fault_notifier.get_current_faults()

        # 通知済み故障関節を取得
        notified_faults = env.scene["robot"]._fault_notifier.get_notified_faults()
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.joint_list_right = [
            "right_hip_yaw",
            "right_hip_roll",
            "right_hip_pitch",
            "right_knee",
        ]
        self.joint_torque_right = {
            1: 2.0,
            4: 10.0,
            8: 20.0,
            12: 50.0,
        }

        self.joint_list_left = [
            "left_hip_yaw",
            "left_hip_roll",
            "left_hip_pitch",
            "left_knee",
        ]
        self.joint_torque_left = {
            0: 2.0,
            3: 10.0,
            7: 20.0,
            11: 50.0,
        }

        self.right_legs = SceneEntityCfg()
        self.right_legs.name = "robot"
        self.right_legs.joint_names = self.joint_list_right
        self.right_legs.resolve(env.scene)

        self.left_legs = SceneEntityCfg()
        self.left_legs.name = "robot"
        self.left_legs.joint_names = self.joint_list_left
        self.left_legs.resolve(env.scene)

        torque_300 = [300.0 for _ in range(len(self.right_legs.joint_ids + self.left_legs.joint_ids))]
        self.torque_300_torch = torch.tensor(torque_300, device="cuda")
        self.log_files = [None for _ in range(env.num_envs)]
        self.asset = env.scene["robot"]
        self.classifier = EnvIdClassifier(env.num_envs)

        # 遅延通知用のnotifierを初期化
        delay_range = cfg.params.get("delay_range", (5, 20))
        self.fault_notifier = DelayedFaultNotifier(env.num_envs, delay_range=delay_range)

        # envにnotifierへの参照を保存（他のイベントからアクセス可能にする）
        env._fault_notifier = self.fault_notifier

        # 故障バッファ（各環境の現在の故障関節ID、-1は故障なし）
        self.current_fault_buffer = torch.full((env.num_envs,), -1, dtype=torch.long, device="cuda")

    def __del__(self):
        for f in self.log_files:
            if f is not None:
                f.close()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        joint_torque: float,
        target_joint_cfg: SceneEntityCfg,
        normal_size: int = 0,
        delay_range: tuple[int, int] = (5, 20),  # 使用はinitで行う
        logging: bool = False
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cuda")
            print("change_random_joint_torque_with_delayed_notification: env_ids is None, so set to all envs")
        else:
            env_ids = env_ids.cuda()

        import random

        robot = env.unwrapped.scene["robot"]
        joint_effort_limits = robot.data.joint_effort_limits
        reseted_limits = joint_effort_limits < (joint_torque + 1)
        reseted_limits = reseted_limits.sum(dim=1) > 0
        not_reseted_limits_env_indices = torch.nonzero(~reseted_limits, as_tuple=False).squeeze(-1)

        if env.num_envs > 1:
            classified_env_ids = self.classifier.classify_by_envid(not_reseted_limits_env_indices)
            right_tensors = [classified_env_ids[i] for i in range(4) if len(classified_env_ids[i]) > 0]
            left_tensors = [classified_env_ids[i] for i in range(4, 8) if len(classified_env_ids[i]) > 0]
            env_ids_right = torch.cat(right_tensors) if len(right_tensors) > 0 else torch.tensor([], dtype=torch.long, device="cuda")
            env_ids_left = torch.cat(left_tensors) if len(left_tensors) > 0 else torch.tensor([], dtype=torch.long, device="cuda")

            # 右側をランダム選出
            target_joint_right = [joint_id for joint_id in target_joint_cfg.joint_ids if joint_id in self.right_legs.joint_ids]
            random_joint_right = random.choice(target_joint_right + [999 for _ in range(normal_size)])
            if random_joint_right != 999:
                joint_torque_torch = torch.tensor(self.joint_torque_right[random_joint_right], device="cuda")
                if len(env_ids_right) > 0:
                    self.asset.write_joint_effort_limit_to_sim(joint_torque_torch, [random_joint_right], env_ids_right)
                    # 故障バッファに書き込み
                    self.current_fault_buffer[env_ids_right] = random_joint_right
                    # 遅延通知用に登録
                    self.fault_notifier.register_fault(env_ids_right, random_joint_right)

            # 左側をランダム選出
            target_joint_left = [joint_id for joint_id in target_joint_cfg.joint_ids if joint_id in self.left_legs.joint_ids]
            random_joint_left = random.choice(target_joint_left + [999 for _ in range(normal_size)])
            if random_joint_left != 999:
                joint_torque_torch = torch.tensor(self.joint_torque_left[random_joint_left], device="cuda")
                if len(env_ids_left) > 0:
                    self.asset.write_joint_effort_limit_to_sim(joint_torque_torch, [random_joint_left], env_ids_left)
                    # 故障バッファに書き込み
                    self.current_fault_buffer[env_ids_left] = random_joint_left
                    # 遅延通知用に登録
                    self.fault_notifier.register_fault(env_ids_left, random_joint_left)
        else:
            # 単一環境の場合
            target_joint_ids = target_joint_cfg.joint_ids
            single_random_joint = random.choice(target_joint_ids + [999 for _ in range(normal_size)])
            if single_random_joint != 999:
                if single_random_joint in self.joint_torque_right:
                    joint_torque = self.joint_torque_right[single_random_joint]
                elif single_random_joint in self.joint_torque_left:
                    joint_torque = self.joint_torque_left[single_random_joint]
                joint_torque_torch = torch.tensor(joint_torque, device="cuda")
                self.asset.write_joint_effort_limit_to_sim(joint_torque_torch, [single_random_joint], env_ids)
                # 故障バッファに書き込み
                self.current_fault_buffer[env_ids] = single_random_joint
                # 遅延通知用に登録
                self.fault_notifier.register_fault(env_ids, single_random_joint)

        if logging:
            for idx in env_ids_left.tolist():
                if self.log_files[idx] is None:
                    self.log_files[idx] = open(f"./nn_data/joint_torque_event_log_env_{idx}.dat", "w")
                    self.log_files[idx].write("common_step_counter,target_joint_id\n")
                if random_joint_left != 999:
                    self.log_files[idx].write(f"{env.common_step_counter},{random_joint_left}\n")
            for idx in env_ids_right.tolist():
                if self.log_files[idx] is None:
                    self.log_files[idx] = open(f"./nn_data/joint_torque_event_log_env_{idx}.dat", "w")
                    self.log_files[idx].write("common_step_counter,target_joint_id\n")
                if random_joint_right != 999:
                    self.log_files[idx].write(f"{env.common_step_counter},{random_joint_right}\n")

    def get_fault_buffer(self) -> torch.Tensor:
        """各環境の現在の故障関節IDを取得する"""
        return self.current_fault_buffer.clone()


class sample_fault_notification_delay(ManagerTermBase):
    """
    エピソード開始時に故障通知の遅延ステップ数をサンプリングするリセットイベント。

    このイベントは、change_random_joint_torque_with_delayed_notificationと
    組み合わせて使用する。

    使用方法:
        EventCfgにて以下のように設定:

        sample_fault_notification_delay = EventTerm(
            func=mdp.sample_fault_notification_delay,
            mode="reset",
            params={}
        )
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.env = env

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cuda")
        else:
            env_ids = env_ids.cuda()

        # envに保存されたfault_notifierを取得してリセット
        if hasattr(env, '_fault_notifier') and env._fault_notifier is not None:
            env._fault_notifier.on_episode_reset(env_ids)


class reset_fault_buffer(ManagerTermBase):
    """
    エピソード開始時に故障バッファをリセットするイベント。

    change_random_joint_torque_with_delayed_notificationと組み合わせて使用する。

    使用方法:
        EventCfgにて以下のように設定:

        reset_fault_buffer = EventTerm(
            func=mdp.reset_fault_buffer,
            mode="reset",
            params={}
        )
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cuda")
        else:
            env_ids = env_ids.cuda()

        # change_random_joint_torque_with_delayed_notificationのインスタンスを探して
        # バッファをリセットする
        if hasattr(env, 'event_manager'):
            for term_cfgs in env.event_manager._mode_class_term_cfgs.values():
                for term_cfg in term_cfgs:
                    if isinstance(term_cfg.func, change_random_joint_torque_with_delayed_notification):
                        term_cfg.func.current_fault_buffer[env_ids] = -1
                        return