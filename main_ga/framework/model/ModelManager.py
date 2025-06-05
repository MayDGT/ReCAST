import time

import numpy as np

from stable_baselines3 import SAC


class NPCModel:
    model_list:{}
    __instance = None

    def __init__(self):
        model = {}
        model['angry_junction'] = SAC.load('/main_ga/model/junction/angry_junction.zip')
        model['cautious_junction'] = SAC.load('/main_ga/model/junction/cautious_junction.zip')
        model['hesitant_junction'] = SAC.load('/main_ga/model/junction/hesitant_junction.zip')

        model['angry_straight'] = SAC.load('/main_ga/model/straight/angry_straight.zip')
        model['cautious_straight'] = SAC.load('/main_ga/model/straight/cautious_straight.zip')
        model['hesitant_straight'] = SAC.load('/main_ga/model/straight/hesitant_straight.zip')
        self.model_list= model
        NPCModel.__instance = self

    @staticmethod
    def get_instance() -> 'NPCModel':
        return NPCModel.__instance

if __name__ == '__main__':
    print(time.time())
    m = NPCModel()
    print(time.time())
    model = NPCModel.get_instance()
    print(time.time())
    action = model.model_list['angry_straight'].predict(np.array([1,1,1,1,1,1]))
    print(action[0]) #[-5.8518896e+00 -4.3351972e-03]
    print(time.time())