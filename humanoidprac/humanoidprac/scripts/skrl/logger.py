from isaaclab.envs import ManagerBasedRLEnvCfg
import json

class SettingLogger():
    def __init__(self,env_cfg:ManagerBasedRLEnvCfg):
        print("Set up joint config logger...")
        self.env_cfg = env_cfg

    def log_setting(self,filepath):
        joint_cfg = self.env_cfg.events.change_joint_torque
        data = {
            "joint_names": joint_cfg.params["asset_cfg"].joint_names,
            "joint_torques": joint_cfg.params["joint_torque"]
        }
        self.write_to_json(data,filepath)

    def write_to_json(self,data:dict,filepath):
        with open(filepath,'w',encoding='utf-8') as f:
            json.dump(data,f,indent=2)
        return 