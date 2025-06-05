

import time
import threading
from logging import Logger
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon, Point

from apollo.ApolloContainer import ApolloContainer
from apollo.ApolloRunner import ApolloRunner
from apollo.CyberBridge import Topics
from framework.scenario import Scenario
from framework.scenario.TrafficControlManager import TrafficControlManager
from hdmap.MapParser import MapParser
from modules.perception.proto.perception_obstacle_pb2 import PerceptionObstacles
from modules.common.proto.header_pb2 import Header
from apollo.utils import clean_apollo_dir, pointenu_to_point, generate_adc_polygon, localization_to_obstacle, \
    obstacle_to_polygon
from config import SCENARIO_UPPER_LIMIT, HD_MAP_PATH
from utils import save_record_files_and_chromosome


class ScenarioRunner:
    container: ApolloContainer
    runner: ApolloRunner
    curr_scenario: Optional[Scenario]
    tm: Optional[TrafficControlManager]
    is_initialized: bool
    map: MapParser
    __instance = None

    def __init__(self, container: ApolloContainer) -> None:
        self.container = container
        self.curr_scenario = None
        self.is_initialized = False
        self.map = MapParser(HD_MAP_PATH)
        ScenarioRunner.__instance = self

    @staticmethod
    def get_instance() -> 'ScenarioRunner':
        return ScenarioRunner.__instance

    def set_scenario(self, s: Scenario):
        self.curr_scenario = s
        self.is_initialized = False

    def init_scenario(self):
        """
        Initialize the scenario
        """
        adc = self.curr_scenario.ad_section.adc
        self.runner = ApolloRunner(
            nid=0,
            ctn=self.container,
            start=adc.initial_position,
            start_time=adc.start_t,
            waypoints=adc.waypoints
        )

        self.runner.initialize()
        clean_apollo_dir()
        self.tm = TrafficControlManager(self.curr_scenario.tc_section)
        self.is_initialized = True

    def run_scenario(self, generation_name: str, scenario_name: str, save_record=False):

        runner_time = 0
        npcs = self.curr_scenario.npc_section
        header_sequence_num = 0

        if save_record:
            self.runner.container.start_recorder(scenario_name)

        print("excutatio  start")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        while True:

            if self.runner.should_send_routing(runner_time / 1000):
                self.runner.send_routing()
            tld = self.tm.get_traffic_configuration(runner_time / 1000)

            polygons = dict()
            localizations = dict()
            speeds = dict()

            location = self.runner.localization
            obstacle = localization_to_obstacle(0, location)
            adc_polygon = obstacle_to_polygon(obstacle)
            polygons[self.runner.nid] = adc_polygon
            localizations[self.runner.nid] = Point(location.pose.position.x, location.pose.position.y)
            for npc in npcs.npcs:
                curr_head = npc.last_state[-1]['heading']
                curr_pos = npc.last_state[-1]['position']  # PointENU
                point_now = pointenu_to_point(curr_pos)
                points = generate_adc_polygon(curr_pos, curr_head)
                curr_polygon = Polygon([[x.x, x.y] for x in points])
                polygons[npc.nid] = curr_polygon
                localizations[npc.nid] = point_now
                speeds[npc.nid] = npc.last_state[-1]['speed']
            ad = polygons[self.runner.nid]
            ob = [polygons[x] for x in polygons if x != self.runner.nid]
            for o in ob:
                self.runner.set_min_distance(ad.distance(o))

            observations = dict()
            infos = dict()

            for npc in npcs.npcs:
                road_type = npc.road
                point_now = localizations[npc.nid]
                curr_polygon = polygons[npc.nid]
                curr_head = npc.last_state[-1]['heading']
                currentLane = self.map.find_lane(point_now, npc.routing)
                agent_index = npc.routing.index(currentLane)

                speed = npc.last_state[-1]['speed']

                priorityCarAhead = dict()
                nearestCarAhead = -1
                nearestCarSpeed = -1
                for id, l in localizations.items():
                    if id == npc.nid: continue
                    minlane, dist = self.map.isin_lanes(l, npc.routing)
                    if (dist < 1):
                        id_index = npc.routing.index(minlane)
                        if (id_index > agent_index):
                            priorityCarAhead[id] = curr_polygon.distance(polygons[id])
                        elif (id_index == agent_index):
                            if (self.map.get_postion_relation(curr_head, l, point_now)):
                                priorityCarAhead[id] = curr_polygon.distance(polygons[id])
                    else:
                        continue
                if priorityCarAhead:
                    nearetNPC = min(priorityCarAhead, key = priorityCarAhead.get)
                    nearestCarAhead = priorityCarAhead[nearetNPC]
                    nearestCarSpeed = speeds[nearetNPC]

                # routeComplete -- number
                point_start = pointenu_to_point(npc.start[0])
                route_complete = point_now.distance(point_start) / 20

                if(road_type == 'straight'):
                    junctionAhead = 100
                    junc_polygon = dict()
                    junctions_id = self.map.get_junctions()
                    for junc_id in junctions_id:
                        poly = self.map.get_junction_polygon(junc_id)
                        poly_point = Point(poly.centroid.x, poly.centroid.y)
                        if (self.map.get_postion_relation(curr_head, poly_point, point_now)):
                            junc_polygon[junc_id] = point_now.distance(poly)
                    if junc_polygon:
                        junctionAhead = min(junc_polygon.values())

                    leftlane = None
                    boundaryAllow = False
                    carLeftLane = -1
                    carRightLane = -1
                    distanceToLeftLane = -1
                    distanceToOriginalLane = -1
                    permit_list = [1, 2, 4]
                    boundaryLeft = self.map.get_lane_boundary_by_id(currentLane)['left']
                    if (boundaryLeft in permit_list):
                        boundaryAllow = True
                        leftlane = self.map.get_left_neighbor_forward_lane(currentLane)
                        distanceToLeftLane = point_now.distance(self.map.get_lane_central_curve(leftlane))
                        distanceToOriginalLane = point_now.distance(self.map.get_lane_central_curve(currentLane))

                    allow_to_leftchange = junctionAhead > 30 and boundaryAllow
                    if (allow_to_leftchange):
                        carLeftAhead = dict()
                        for id, l in localizations.items():
                            minlane, dist = self.map.isin_lanes(l, [leftlane])
                            if (dist < 1 and self.map.get_postion_relation(curr_head, l, point_now)):
                                carLeftAhead[id] = curr_polygon.distance(polygons[id])
                        if carLeftAhead:
                            carLeftLane = min(carLeftAhead.values())

                    allow_to_rightchange = junctionAhead > 30
                    if (allow_to_rightchange):
                        carRightAhead = dict()
                        for id, l in localizations.items():
                            minlane, dist = self.map.isin_lanes(l, [currentLane])
                            if (dist < 1 and self.map.get_postion_relation(curr_head, l, point_now)):
                                carRightAhead[id] = curr_polygon.distance(polygons[id])
                        if carRightAhead:
                            carRightLane = min(carRightAhead.values())

                    if distanceToLeftLane > 10: distanceToLeftLane = 10
                    if distanceToOriginalLane > 10: distanceToOriginalLane = 10


                    ob = np.array([
                        speed,
                        nearestCarAhead,
                        nearestCarSpeed,
                        carLeftLane,
                        carRightLane,
                        distanceToLeftLane,
                        distanceToOriginalLane,
                        route_complete
                    ])
                    info = {
                        'road': 'straight'
                    }

                else:
                    signalAhead = None
                    stopSignAhead = None
                    stopLineAhead = 100
                    # signal
                    signal = self.map.get_signal_control_lane(currentLane)
                    if signal != "":
                        signal_color = None
                        for i in range(0, 15):
                            if signal == tld.traffic_light[i].id:
                                signal_color = tld.traffic_light[i].color
                                break
                        signalAhead = signal_color
                        stopline = self.map.get_stopline_of_signal(signal)
                        stopLineAhead = point_now.distance(stopline)
                    # stop_sign
                    stop_sign = self.map.get_sign_control_lane(currentLane)
                    if stop_sign != "":
                        stopSignAhead = True
                        stopline = self.map.get_stopline_of_stop(stop_sign)
                        stopLineAhead = point_now.distance(stopline)

                    ob = np.array([
                        speed,
                        nearestCarAhead,
                        stopLineAhead,
                        stopSignAhead,
                        signalAhead,
                        route_complete
                    ])
                    info = {
                        'road': 'junction'
                    }

                observations[npc.nid] = ob
                infos[npc.nid] = info

            obs = npcs.get_obstacles(runner_time/1000, observations, infos)

            header = Header(
                timestamp_sec=time.time(),
                module_name='MAGGIE',
                sequence_num=header_sequence_num
            )
            bag = PerceptionObstacles(
                header=header,
                perception_obstacle=obs,
            )
            self.container.bridge.publish(
                Topics.Obstacles, bag.SerializeToString())
            self.container.bridge.publish(
                Topics.TrafficLight, tld.SerializeToString()
            )
            header_sequence_num += 1
            if runner_time / 1000 >= SCENARIO_UPPER_LIMIT:
                break
            time.sleep(0.1)
            runner_time += 100

        print("executation end")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        if save_record:
            # 为所有npc的style创建字典
            style = dict()
            style_config = dict()
            style_robust = dict()
            for npc in self.curr_scenario.npc_section.npcs:
                style_config[npc.nid] = npc.style
                style_robust[npc.nid] = npc.reward.reward_list
            style['config'] = style_config
            style['robustness'] = style_robust

            self.runner.container.stop_recorder()
            time.sleep(2)
            save_record_files_and_chromosome(
                generation_name, scenario_name, self.curr_scenario.to_dict(), style)

        self.runner.stop('MAIN')

        return self.runner, self.curr_scenario.ad_section.adc, self.curr_scenario.npc_section.npcs


    def replay_scenario(self):
        runner_time = 0
        npcs = self.curr_scenario.npc_section
        # 每次run_scenario都会清0，每0.1s会加一
        header_sequence_num = 0

        while True:
            if self.runner.should_send_routing(runner_time / 1000):
                self.runner.send_routing()
            tld = self.tm.get_traffic_configuration(runner_time / 1000)
            obs = npcs.get_obstacles_replay(runner_time / 1000)
            header = Header(
                timestamp_sec=time.time(),
                module_name='MAGGIE',
                sequence_num=header_sequence_num
            )
            bag = PerceptionObstacles(
                header=header,
                perception_obstacle=obs,
            )
            self.container.bridge.publish(
                Topics.Obstacles, bag.SerializeToString())
            self.container.bridge.publish(
                Topics.TrafficLight, tld.SerializeToString()
            )
            header_sequence_num += 1
            if runner_time / 1000 >= SCENARIO_UPPER_LIMIT:
                break
            time.sleep(0.1)
            runner_time += 100



        self.runner.stop('MAIN')
