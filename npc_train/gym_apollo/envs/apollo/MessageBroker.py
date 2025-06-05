import time
import logging
import threading
from typing import List
from logging import Logger
from threading import Thread
from shapely.geometry import Polygon
from gym_apollo.envs.framework.scenario.CAgent import CAgent
from gym_apollo.envs.modules.common.proto.header_pb2 import Header
from gym_apollo.envs.modules.perception.proto.perception_obstacle_pb2 import PerceptionObstacles
from gym_apollo.envs.apollo.utils import localization_to_obstacle, obstacle_to_polygon
from gym_apollo.envs.apollo.CyberBridge import Channel, Topics
from gym_apollo.envs.framework.scenario.PedestrianManager import PedestrianManager
from gym_apollo.envs.utils import get_logger
from config import PERCEPTION_FREQUENCY
from gym_apollo.envs.apollo.ApolloRunner import ApolloRunner


class MessageBroker:
    """
    Class to represent MessageBroker, it tracks location of each ADS instance and broadcasts
    perception message to all ADS instances

    :param List[ApolloRunner] runners: list of runners each controlling an ADS instance
    """

    runners: List[ApolloRunner]
    agent:CAgent
    spinning: bool
    logger: Logger
    t: Thread
    action:List#[acc,steer]
    localizations:dict


    def __init__(self, runners: List[ApolloRunner], agent:CAgent) -> None:
        """
        Constructor
        """
        self.runners = runners
        self.spinning = False
        self.logger = self.get_test_logger()
        self.agent = agent
        self.action = [0,0]
        self.localizations = {}
        self.polygons = {}
        self.speeds = {}

    def get_test_logger(self):
        logger = logging.getLogger('MessageBroker')
        logger.setLevel(level=logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        file_handler = logging.FileHandler('test_messagebroker.log', 'w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger
    def broadcast(self, channel: Channel, data: bytes):
        for runner in self.runners:
            runner.container.bridge.publish(channel, data)

    def _spin(self):

        header_sequence_num = 0
        curr_time = 0.0

        while self.spinning:
            locations = dict()
            for runner in self.runners:
                loc = runner.localization
                if loc and loc.header.module_name == 'SimControl':
                    locations[runner.nid] = runner.localization
                    self.localizations[runner.nid] = runner.localization
                    self.speeds[runner.nid] = runner.speeds

            obs = dict()
            obs_poly = dict()

            for k in locations:

                obs[k] = localization_to_obstacle(k, locations[k])
                obs_poly[k] = obstacle_to_polygon(obs[k])
                self.polygons[k] = obs_poly[k]

            pm = PedestrianManager.get_instance()
            pds = pm.get_pedestrians(curr_time)

            acc = self.action[0]
            steer = self.action[1]
            curr_state = {'acc':acc, 'steer':steer, 'time':curr_time}
            agent = self.agent.get_obstacles(curr_state)

            # publish obstacle to all running instances
            for runner in self.runners:
                perception_obs = [obs[x] for x in obs if x != runner.nid] + pds +agent
                header = Header(
                    timestamp_sec=time.time(),
                    module_name='MAGGIE',
                    sequence_num=header_sequence_num
                )
                bag = PerceptionObstacles(
                    header=header,
                    perception_obstacle=perception_obs,
                )
                runner.container.bridge.publish(
                    Topics.Obstacles, bag.SerializeToString()
                )

            header_sequence_num += 1
            time.sleep(1/PERCEPTION_FREQUENCY)
            curr_time += 1/PERCEPTION_FREQUENCY

    def spin(self):
        """
        Starts to forward localization
        """
        self.logger.debug('Starting to spin')
        if self.spinning:
            return
        self.t = Thread(target=self._spin)
        self.spinning = True
        self.t.start()

    def stop(self):
        """
        Stops forwarding localization
        """
        self.logger.debug('Stopping')
        if not self.spinning:
            return
        self.spinning = False
        self.t.join()


    def set_action(self, action):
        self.action = action

    def get_localization(self):
        return self.localizations

    def get_polygon(self):
        return self.polygons

    def get_agent(self):
        return self.agent

    def get_speed(self):
        return self.speeds
