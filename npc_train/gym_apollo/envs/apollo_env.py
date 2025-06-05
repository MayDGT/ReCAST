import copy
import logging
import math
import time

import gym
import os
import numpy as np
import json
import threading
from gym import spaces
from gym_apollo.envs.apollo.ApolloContainer import ApolloContainer
from gym_apollo.envs.apollo.ApolloRunner import ApolloRunner
from gym_apollo.envs.apollo.MessageBroker import MessageBroker
from gym_apollo.envs.apollo.CyberBridge import Topics
from gym_apollo.envs.apollo.utils import generate_adc_polygon, clean_apollo_dir, pointenu_to_point
from gym_apollo.envs.framework.scenario.CAgent import CAgent
from gym_apollo.envs.framework.scenario.PedestrianManager import PedestrianManager
from gym_apollo.envs.framework.scenario.TrafficControlManager import TrafficControlManager
from gym_apollo.envs.hdmap.MapParser import MapParser
from gym_apollo.envs.utils import random_numeric_id
from gym_apollo.envs.framework.scenario import Scenario
from shapely.geometry import Polygon, Point

from config import APOLLO_ROOT, HD_MAP_PATH, MAX_ADC_COUNT
import rtamt
import sys

class ApolloEnv(gym.Env):
    def __init__(self):
        # action[acc, steer]
        params = {
            'destination': 'straight',
            # action space
            'continuous_accel_range': [-5.0, 5.0],  # continuous acceleration range
            'continuous_steer_range': [-0.01, 0.01],  # continuous steering angle range

        }
        self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0],params['continuous_steer_range'][0]]),
                                       np.array([params['continuous_accel_range'][1],params['continuous_steer_range'][1]]), dtype=np.float32)  # acc, steer

        # self.observation_space = spaces.Box(np.array([params['dist_destination'][0],params['lane_offset'][0],params['orientation_offset'][0]]),
        #                                     np.array([params['dist_destination'][1],params['lane_offset'][1],params['orientation_offset'][1]]), dtype=np.float32)

        self.observation_space = spaces.Box(np.array([-10, -10, -10, -10, -10, -10, -10]), np.array([10, 10, 10, 10, 10, 10, 10]), dtype=np.float32)

        self.save_record = True
        self.runner_time = 0
        self.scenario_id = -1
        self.runners = list()
        self.mbk = None
        self.map = MapParser(HD_MAP_PATH)
        print("the map is ok")
        self.agent_dest = params['destination']
        self.scenario = None
        self.scenario_name = 'None'
        self.sp = rtamt.STLSpecification()


        self.sp.spec = ('always[0,30](((nearestCarAhead<10 and nearestCarSpeed<1 and carLeftLane>20) implies (eventually[0,5](speed>11)))'
                        '               and ((nearestCarAhead<10 and nearestCarSpeed<1 and carLeftLane>20)implies(eventually[0,5](distanceToLeftLane<1))) '
                        '               and ((distanceToLeftLane<distanceToOriginalLane and carRightLane>20)implies(eventually[0,5](speed>11))) '
                        '               and ((distanceToLeftLane<distanceToOriginalLane and carRightLane>20)implies(eventually[0,5](distanceToOriginalLane<1)))'
                        '               and (speed < 13)'
                        '               and ((nearestCarAhead<10)implies(eventually[0,3](speed<1))))'
                        'and '
                        'eventually[0,30](route_complete>10)'
                        )
        self.sp.unit = 's'
        self.sp.set_sampling_period(100, 'ms', 0.1)
        self.initialized = 0
        print("init over")

    def init_scenario(self):
        nids = random_numeric_id(len(self.curr_scenario.ad_section.adcs))
        self.runners = list()
        for i, c, a in zip(nids, self.containers, self.curr_scenario.ad_section.adcs):
            a.apollo_container = c.container_name
            self.runners.append(
                ApolloRunner(
                    nid=i,
                    ctn=c,
                    start=a.initial_position,
                    waypoints=a.waypoints,
                    start_time=a.start_t
                )
            )

        threads = list()
        for index in range(len(self.runners)):
            threads.append(threading.Thread(
                target=self.runners[index].initialize
            ))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        clean_apollo_dir()
        self.pm = PedestrianManager(self.curr_scenario.pd_section)
        self.tm = TrafficControlManager(self.curr_scenario.tc_section)

    def step(self, action):
        # print(action)
        self.mbk.set_action(action)
        runner_time = self.runner_time

        # Publish TrafficLight
        tld = self.tm.get_traffic_configuration(runner_time / 1000)
        self.mbk.broadcast(Topics.TrafficLight, tld.SerializeToString())

        # Send Routing
        for ar in self.runners:
            if ar.should_send_routing(runner_time / 1000):
                ar.send_routing()

        ls = self.mbk.get_localization()
        localizations = dict()
        for id, l in ls.items():
            localizations[id] = Point(l.pose.position.x, l.pose.position.y)
        polygons = self.mbk.get_polygon()
        agent = self.mbk.get_agent()
        speeds = self.mbk.get_speeds()
        start_pos = agent.start[0]
        start_head = agent.start[1]
        dest_pos = agent.destination[0]
        dest_head = agent.destination[1]
        curr_pos = agent.last_state['position'] #PointENU
        curr_head = agent.last_state['heading']
        point_now = pointenu_to_point(curr_pos)
        points = generate_adc_polygon(curr_pos, curr_head)
        curr_polygon = Polygon([[x.x, x.y] for x in points])

        speed = agent.last_state['speed']
        currentLane = self.map.find_lane(point_now, agent.routing)
        agent_index = agent.routing.index(currentLane)
        currenrTurn = self.map.get_lane_turn_by_id(currentLane)

        priorityCarAhead = dict()
        nearestCarAhead = 100
        nearestCarSpeed = 100
        for id, l in localizations.items():
            minlane, dist = self.map.isin_lanes(l, agent.routing)
            if(dist < 1):
                id_index = agent.routing.index(minlane)
                if(id_index > agent_index):
                    priorityCarAhead[id] = curr_polygon.distance(polygons[id])
                elif(id_index == agent_index):
                    if(self.map.get_postion_relation(curr_head, l, point_now)):
                        priorityCarAhead[id] = curr_polygon.distance(polygons[id])
            else: continue
        if priorityCarAhead:
            nearetNPC = min(priorityCarAhead, key=priorityCarAhead.get)
            nearestCarAhead = priorityCarAhead[nearetNPC]
            nearestCarSpeed = speeds[nearetNPC]

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
        carLeftLane = False
        carRightLane = False
        distanceToLeftLane = 0
        distanceToOriginalLane = 0
        permit_list = [1, 2, 4]
        boundaryLeft = self.map.get_lane_boundary_by_id(currentLane)['left']
        if (boundaryLeft in permit_list):
            leftlane = self.map.get_left_neighbor_forward_lane(currentLane)
            distanceToLeftLane = point_now.distance(self.map.get_lane_central_curve(leftlane))
            distanceToOriginalLane = point_now.distance(self.map.get_lane_central_curve(currentLane))

        # need_to_leftchange
        need_to_leftchange = nearestCarAhead < 10 and distanceToLeftLane > distanceToOriginalLane
        if (need_to_leftchange):
            need_to_leftchange = 10
        else:
            need_to_leftchange = 0
        # allow to leftchange
        allow_to_leftchange = junctionAhead > 30 and (boundaryLeft in permit_list)
        if(allow_to_leftchange):
            carLeftAhead = dict()
            for id, l in localizations.items():
                minlane, dist = self.map.isin_lanes(l, [leftlane])
                if (dist < 1 and self.map.get_postion_relation(curr_head, l, point_now)):
                    carLeftAhead[id] = curr_polygon.distance(polygons[id])
            if carLeftAhead and min(carLeftAhead.values()) < 5:
                carLeftLane = True
        allow_to_leftchange = allow_to_leftchange and not carLeftLane
        toLeftChanging = need_to_leftchange and allow_to_leftchange
        if(toLeftChanging):agent.change_state('left', True)
        if (toLeftChanging):
            toLeftChanging = 10
        else:
            toLeftChanging = 0

        # need_to_rightchange
        need_to_rightchange = distanceToLeftLane < distanceToOriginalLane
        # allow to rightchange
        allow_to_rightchange = junctionAhead > 30
        if(allow_to_rightchange and need_to_rightchange):
            carRightAhead = dict()
            for id, l in localizations.items():
                minlane, dist = self.map.isin_lanes(l, [currentLane])
                if (dist < 1 and self.map.get_postion_relation(curr_head, l, point_now)):
                    carRightAhead[id] = curr_polygon.distance(polygons[id])
            if carRightAhead and min(carRightAhead.values()) < 5:
                carRightLane = True
        allow_to_rightchange = allow_to_rightchange and not carRightLane
        toRightChanging = need_to_rightchange and allow_to_rightchange
        if (toRightChanging): agent.change_state('right', True)
        if (toRightChanging):
            toRightChanging = 10
        else:
            toRightChanging = 0


        need_to_stop = nearestCarAhead < 5
        if (need_to_stop):
            need_to_stop = 10
        else:
            need_to_stop = 0
        point_start = pointenu_to_point(start_pos)
        route_complete = point_now.distance(point_start) / 20
        obs = np.array([speed,
                        nearestCarAhead,
                        nearestCarSpeed,
                        carLeftLane,
                        carRightLane,
                        distanceToLeftLane,
                        distanceToOriginalLane,
                        route_complete])

        done = False

        if runner_time==30000:
            done = True
            stop_reason = 'reach_time_limit'
            print(stop_reason)



        info = {
            'need_to_leftchange':need_to_leftchange,
            'need_to_rightchange':need_to_rightchange,
            'allow_to_leftchange':allow_to_leftchange,
            'allow_to_rightchange':allow_to_rightchange,
            'toLeftChanging':toLeftChanging,
            'toRightChanging':toRightChanging,
            'need_to_stop':need_to_stop,
        }

        # reward
        if(done == False):
            reward = self.get_reward(obs, info)
        else: reward = 0

        self.runner_time += 100
        time.sleep(0.1)
        return (copy.deepcopy(obs), reward, done, copy.deepcopy(info))


    def reset(self):
        if self.mbk is not None:
            self.mbk.stop()
            for runner in self.runners:
                runner.stop('MAIN')

        self.scenario_id += 1
        self.runner_time = 0
        self.curr_scenario = Scenario.get_one_for_agent(self.agent_dest)
        self.record_scenario("scenario.json")
        self.agent = self.curr_scenario.ag_section
        containers = [ApolloContainer(APOLLO_ROOT, f'ROUTE_{x}') for x in range(MAX_ADC_COUNT)]
        self.containers = containers
        for ctn in containers:
            print(f"start:{ctn.username}")
            ctn.start_instance()
            # ctn.start_dreamview()
        self.init_scenario()
        self.mbk = MessageBroker(runners=self.runners, agent= self.agent)
        self.mbk.spin()
        start_point = pointenu_to_point(self.agent.start[0])
        dest_point = pointenu_to_point(self.agent.destination[0])
        return np.array([0,0,0,0,0,0,0])

    def render(self, mode="human"):
        pass
    def seed(self, seed=None):
        pass

    def change_time(self, time):
        self.runner_time = time

    def record_scenario(self, name):
        current_dir = os.getcwd()
        dest_file = os.path.join(current_dir, name)
        print(dest_file)
        with open(dest_file, 'w') as fp:
            json.dump(self.curr_scenario.to_dict(), fp, indent=4)


    def get_reward(self, obs, info):
        speed = obs[0]
        nearestCarAhead = obs[1]
        nearestCarSpeed = obs[2]
        carLeftLane = obs[3]
        carRightLane = obs[4]
        distanceToLeftLane = obs[5]
        distanceToOriginalLane = obs[6]
        route_complete = obs[7]
        time_step = round(self.runner_time/1000, 1)
        if self.initialized == 0:
            self.sp.declare_var('speed', 'float')
            self.sp.declare_var('nearestCarAhead', 'float')
            self.sp.declare_var('nearestCarSpeed', 'float')
            self.sp.declare_var('carLeftLane', 'float')
            self.sp.declare_var('carRightLane', 'float')
            self.sp.declare_var('distanceToLeftLane', 'float')
            self.sp.declare_var('distanceToOriginalLane', 'float')
            self.sp.declare_var('route_complete', 'float')
            self.initialized = 1
            try:
                self.sp.parse()
                self.sp.pastify()
            except rtamt.RTAMTException as err:
                print('RTAMT Exception: {}'.format(err))
                sys.exit()

        rob = self.sp.update(time_step, [('speed', speed),
                                         ('nearestCarAhead', nearestCarAhead),
                                         ('nearestCarSpeed', nearestCarSpeed),
                                         ('carLeftLane', carLeftLane),
                                         ('carRightLane', carRightLane),
                                         ('distanceToLeftLane', distanceToLeftLane),
                                         ('distanceToOriginalLane', distanceToOriginalLane),
                                         ('route_complete', route_complete)])
        return rob

    def stop(self):
        if self.mbk is not None:
            self.mbk.stop()
            print("the message broker is stop")
            for runner in self.runners:
                runner.stop('MAIN')
            print("the apollo runners are stop")