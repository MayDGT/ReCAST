from gym_apollo.envs.framework.scenario import Scenario

import time
import threading
from logging import Logger
from typing import List, Optional, Tuple

from gym_apollo.envs.apollo import ApolloContainer
from gym_apollo.envs.apollo.ApolloRunner import ApolloRunner
from gym_apollo.envs.apollo import Topics
from gym_apollo.envs.apollo.MessageBroker import MessageBroker
from gym_apollo.envs.framework.scenario.PedestrianManager import PedestrianManager
from gym_apollo.envs.framework.scenario.TrafficControlManager import TrafficControlManager
from gym_apollo.envs.apollo import clean_appolo_dir
from gym_apollo.envs.utils import get_logger, get_scenario_logger, random_numeric_id, save_record_files_and_chromosome
from gym_apollo.envs.framework.scenario.ad_agents import ADAgent

from config import SCENARIO_UPPER_LIMIT


class ScenarioRunner:
    """
    Executes a scenario based on the specification

    :param List[ApolloContainer] containers: containers to be used for scenario
    """
    logger: Logger
    containers: List[ApolloContainer]
    curr_scenario: Optional[Scenario]
    pm: Optional[PedestrianManager]
    tm: Optional[TrafficControlManager]
    is_initialized: bool
    __instance = None

    __runners: List[ApolloRunner]

    def __init__(self, containers: List[ApolloContainer]) -> None:
        """
        Constructor
        """
        self.logger = get_logger('ScenarioRunner')
        self.containers = containers
        self.curr_scenario = None
        self.is_initialized = False
        ScenarioRunner.__instance = self

    @staticmethod
    def get_instance() -> 'ScenarioRunner':
        """
        Returns the singleton instance

        :returns: an instance of runner
        :rtype: ScenarioRunner
        """
        return ScenarioRunner.__instance

    def set_scenario(self, s: Scenario):
        """
        Set the scenario for this runner

        :param Scenario s: scenario representation
        """
        self.curr_scenario = s
        self.is_initialized = False

    def init_scenario(self):
        """
        Initialize the scenario
        """
        nids = random_numeric_id(len(self.curr_scenario.ad_section.adcs))
        self.__runners = list()
        for i, c, a in zip(nids, self.containers, self.curr_scenario.ad_section.adcs):
            a.apollo_container = c.container_name
            self.__runners.append(
                ApolloRunner(
                    nid=i,
                    ctn=c,
                    start=a.initial_position,
                    waypoints=a.waypoints,
                    start_time=a.start_t
                )
            )

        # initialize Apollo instances
        threads = list()
        for index in range(len(self.__runners)):
            threads.append(threading.Thread(
                target=self.__runners[index].initialize
            ))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # remove Apollo logs
        clean_appolo_dir()

        # initialize pedestrian and traffic control manager
        self.pm = PedestrianManager(self.curr_scenario.pd_section)
        self.tm = TrafficControlManager(self.curr_scenario.tc_section)
        self.is_initialized = True

    def run_scenario(self, generation_name: str, scenario_name: str, save_record=False) -> List[Tuple[ApolloRunner, ADAgent]]:
        """
        Execute the scenario based on the specification

        :param str generation_name: name of the generation
        :param str scenario_name: name of the scenario
        :param bool save_record: whether to save records or not
        """
        num_adc = len(self.curr_scenario.ad_section.adcs)
        self.logger.info(
            f'{num_adc} agents running a scenario G{self.curr_scenario.gid}S{self.curr_scenario.cid}.'
        )
        if self.curr_scenario is None or not self.is_initialized:
            print('Error: No chromosome or not initialized')
            return

        mbk = MessageBroker(self.__runners)
        mbk.spin()

        runner_time = 0
        scenario_logger = get_scenario_logger()
        # starting scenario
        if save_record:
            for r in self.__runners:
                r.container.start_recorder(scenario_name)

        # Begin Scenario Cycle
        while True:
            # Publish TrafficLight
            tld = self.tm.get_traffic_configuration(runner_time/1000)
            mbk.broadcast(Topics.TrafficLight, tld.SerializeToString())

            # Send Routing
            for ar in self.__runners:
                if ar.should_send_routing(runner_time/1000):
                    ar.send_routing()

            # Print Scenario Time
            if runner_time % 100 == 0:
                scenario_logger.info(
                    f'Scenario time: {round(runner_time / 1000, 1)}.')

            # Check if scenario exceeded upper limit
            if runner_time / 1000 >= SCENARIO_UPPER_LIMIT:
                scenario_logger.info('\n')
                break

            time.sleep(0.1)
            runner_time += 100

        if save_record:
            for r in self.__runners:
                r.container.stop_recorder()
            # buffer period for recorders to stop
            time.sleep(2)
            save_record_files_and_chromosome(
                generation_name, scenario_name, self.curr_scenario.to_dict())
        # scenario ended
        mbk.stop()
        for runner in self.__runners:
            runner.stop('MAIN')

        self.logger.debug(
            f'Scenario ended. Length: {round(runner_time/1000, 2)} seconds.')

        self.is_initialized = False

        return list(zip(self.__runners, self.curr_scenario.ad_section.adcs))
