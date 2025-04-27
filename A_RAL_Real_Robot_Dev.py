import time
import numpy as np
import json
from xrocs.common.data_type import Joints
from xrocs.core.config_loader import ConfigLoader
from xrocs.utils.logger.logger_loader import logger
from xrocs.core.station_loader import StationLoader
import cv2
# 导入模型class
# from A_RAL_DEV import DTP_Evaluation
from A_RAL_GR1_DTP_DEV import DTP_Evaluation

class JointInference:
    def __init__(self, config_path = None, model_path = None):
        if config_path == None:
            config_path = "/home/eai/Documents/configuration.toml"
        cfg_loader = ConfigLoader(config_path)
        self.cfg_dict = cfg_loader.get_config()
        station_loader = StationLoader(self.cfg_dict)
        self.robot_station = station_loader.generate_station_handle()
        self.robot_station.connect()
        model_cfg = json.load(open('/home/ps/Dev/bamboo/Pretrain_Model/ral_rebuttal/RAL_GR1_0424/config_dev.json'))
    
        print('inference model loading')
        self.infer_model = DTP_Evaluation(model_cfg)
        self.infer_model.reset()
        # 监听
                # 添加任务映射字典
        self.task_mapping = {
            ord('1'): "open the upper drawer",
            ord('2'): "pick bread and place into drawer",
            ord('3'): "close the upper drawer",
        }
        self.current_task = "open the upper drawer"  # 默认任务
        


    def prepare(self):
        for name, _robot in self.robot_station.get_robot_handle().items():
            home = Joints(self.cfg_dict['robot']['arm']['home'][name],
                          num_of_dofs=len((self.cfg_dict['robot']['arm']['home'][name])))
            _robot.reach_target_joint(home)
        for gripper in self.robot_station.get_gripper_handle().values():
            gripper.open()
        time.sleep(2)
        logger.success('Resetting to home success!')

    def inference(self):
        obs = self.robot_station.get_obs()

        while True:
            cv2.imshow("rgb_vis", cv2.imdecode(obs['images']['left'], cv2.IMREAD_COLOR))
            print(f"\rCurrent task: {self.current_task}", end="")
            key = cv2.waitKey(1) & 0xFF
            if key in self.task_mapping:
                self.current_task = self.task_mapping[key]
                self.infer_model.reset()
                print(f"\nSwitched to task: {self.current_task}")
            elif key == ord('r'): # restart
                self.infer_model.reset()
                self.prepare()
                obs = self.robot_station.get_obs()
                self.current_task = "open the upper drawer"

            # elif key = ord('s') # stop

            action_pred: np.ndarray = self.infer_model.infer(obs,self.current_task,key)
            action_pred = action_pred.cpu().numpy()
            robot_targets = self.robot_station.decompose_action(action_pred)
            obs = self.robot_station.step(robot_targets)


if __name__ == '__main__':
    config_path = '/home/ps/Documents/configuration.toml'
    # inference init
    robot = JointInference(config_path=config_path)
    robot.prepare()
    robot.inference()