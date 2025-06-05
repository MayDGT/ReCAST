import math
from random import random
from typing import List, Tuple
from dataclasses import dataclass

from gym_apollo.envs.apollo.utils import dynamic_obstacle_location_to_obstacle,PositionEstimate, pointenu_to_point
from gym_apollo.envs.framework.scenario.ad_agents import ADAgent
from gym_apollo.envs.modules.common.proto.geometry_pb2 import PointENU
from gym_apollo.envs.modules.perception.proto.perception_obstacle_pb2 import PerceptionObstacle

from gym_apollo.envs.hdmap.MapParser import MapParser
from secrets import choice
from gym_apollo.envs.utils import get_logger
@dataclass
class CAgent:
    """
    # 这个类更改为agent的类
    A simplified modeling of constant speed obstacles

    :param List[ADAgent] obs: list of obstacles to be managed
    :nids List[int]: list of ids
    """
    dest: str
    driving_time: float
    last_state: dict
    action: dict
    start: Tuple[PointENU, float]
    destination:Tuple[PointENU, float]
    s_lane: str
    s_s: float
    d_lane: str
    d_s: float
    routing: List[str]
    changing:dict

    @staticmethod
    def get_one(dest: str) -> 'CAgent':
        ma = MapParser.get_instance()
        allowed_lanes = list(ma.get_lanes_not_in_junction())
        while True:
            choice_lane = choice(allowed_lanes)
            routing_ = ma.get_path_from(choice_lane)
            if dest == 'junction':
                allowed_routing = routing_['left'] + routing_['right']
            else:
                allowed_routing = routing_[dest]
            if(len(allowed_routing)!=0): break

        choice_routing = choice(allowed_routing)

        start_length = ma.get_lane_length(choice_routing[0])
        dest_length = ma.get_lane_length(choice_routing[-1])
        start_s = 0
        if start_length > 5:
            start_s = round(random() * (start_length - 5), 1)
        else:
            start_s = 0.0
        dest_s = round(dest_length / 2, 1)

        start_p = ma.get_coordinate_and_heading(choice_routing[0], start_s)
        destination_p = ma.get_coordinate_and_heading(choice_routing[-1], dest_s)
        changing_state = {
            'left': False,
            'right': False
        }

        return CAgent(
            dest = dest,
            driving_time = 0.0,
            last_state= {"position":start_p[0], 'heading':start_p[1], 'speed':0.0, 'time':0.0, 'traveled':0.0},
            action = {'acc':0.0, 'steer':0.0, 'time':0.0},
            start = start_p,
            destination = destination_p,
            s_lane = choice_routing[0],
            s_s = start_s,
            d_lane = choice_routing[-1],
            d_s = dest_s,
            routing = choice_routing,
            changing = changing_state
        )

    @staticmethod
    def get_test_one() -> 'CAgent':
        start_lane = 'lane_1'
        start_s = 10
        dest_lane = 'lane_8'
        dest_s = 10
        routing = ['lane_1', 'lane_34', 'lane_8']
        ma = MapParser.get_instance()
        start_p = ma.get_coordinate_and_heading(start_lane, start_s)
        destination_p = ma.get_coordinate_and_heading(dest_lane, dest_s)
        changing_state = {
            'left': False,
            'right': False
        }
        return CAgent(
            dest='straight',
            driving_time=0.0,
            last_state={"position": start_p[0], 'heading': start_p[1], 'speed': 0.0, 'time': 0.0, 'traveled':0.0},
            action={'acc': 0.0, 'steer': 0.0, 'time': 0.0},
            start=start_p,
            destination=destination_p,
            s_lane=start_lane,
            s_s=start_s,
            d_lane=dest_lane,
            d_s=dest_s,
            routing=routing,
            changing = changing_state
        )


    def calculate_position(self) -> Tuple[PointENU, float, float, float]:
        """
        last_state有：position heading speed time traveled
        curr_state有：acc steer current_time
        返回：位置、朝向、速度
        """
        ma = MapParser.get_instance()
        point_now = pointenu_to_point(self.last_state['position'])
        # 不管在哪个车道正在左转or右转，这个original都是一个车道
        original_lane = ma.find_lane(point_now, self.routing)
        left_lane = ma.get_left_neighbor_forward_lane(original_lane)
        # print(f"origin:{original_lane}, left:{left_lane}, leftcha:{self.changing['left']}, righcha:{self.changing['right']}")
        original_lane_line = ma.get_lane_central_curve(original_lane)
        dist_to_left = 5
        dist_to_origin =5
        if left_lane != None:
            left_lane_line = ma.get_lane_central_curve(left_lane)
            dist_to_left = point_now.distance(left_lane_line)
            dist_to_origin = point_now.distance(original_lane_line)

        if(self.changing['left'] == True):
            if(dist_to_left < 0.5):
                self.changing['left'] = False
                target_lane = current_lane = left_lane
            else:
                target_lane = left_lane
                current_lane = original_lane
        elif(self.changing['right'] == True):
            if (dist_to_origin < 0.5):
                self.changing['right'] = False
                target_lane = current_lane = original_lane
            else:
                target_lane = original_lane
                current_lane = left_lane
        else:
            target_lane = original_lane
            current_lane = original_lane

        _, target_heading = ma.get_lane_and_heading(point_now, current_lane, target_lane)

        curr_heading = target_heading + self.action['steer']
        if(abs(curr_heading) > math.pi):
            if(curr_heading > 0) : curr_heading = curr_heading - 2 * math.pi
            else: curr_heading = curr_heading + 2 * math.pi
        curr_speed = self.last_state['speed'] + self.action['acc'] * self.interval
        if(curr_speed < 0):curr_speed = 0
        dist = curr_speed * self.interval + 0.5 * self.action['acc'] * self.interval**2
        curr_position_x = self.last_state['position'].x + dist * math.cos(curr_heading)
        curr_position_y = self.last_state['position'].y + dist * math.sin(curr_heading)
        traveled = self.last_state['traveled'] + dist
        return (PointENU(x=curr_position_x, y=curr_position_y), curr_heading, curr_speed, traveled)


    def calculate_position_1(self) -> Tuple[PointENU, float, float, float]:

        ma = MapParser.get_instance()
        point_now = pointenu_to_point(self.last_state['position'])
        current_lane = ma.find_lane(point_now, self.routing)
        target_lane = current_lane
        _, target_heading = ma.get_lane_and_heading(point_now, current_lane, target_lane)

        curr_heading = target_heading + self.action['steer']
        if(abs(curr_heading) > math.pi):
            if(curr_heading > 0) : curr_heading = curr_heading - 2 * math.pi
            else: curr_heading = curr_heading + 2 * math.pi
        curr_speed = self.last_state['speed'] + self.action['acc'] * self.interval
        if(curr_speed < 0):curr_speed = 0
        dist = curr_speed * self.interval + 0.5 * self.action['acc'] * self.interval**2
        curr_position_x = self.last_state['position'].x + dist * math.cos(curr_heading)
        curr_position_y = self.last_state['position'].y + dist * math.sin(curr_heading)
        traveled = self.last_state['traveled'] + dist
        return (PointENU(x=curr_position_x, y=curr_position_y), curr_heading, curr_speed, traveled)




    def get_obstacles(self, action:dict) -> PerceptionObstacle:
        """
        Get a list of PerceptionObstacle messages ready to be published

        :param float curr_time: scenario time

        :returns: list of PerceptionObstacle messages
        :rtype: List[PerceptionObstacle]
        """
        # print(action)
        last_state = self.last_state
        self.action = action
        # print(last_state)
        curr_time = action['time']
        self.interval = action['time'] - last_state['time']
        agent_position, agent_heading, agent_speed, traveled= self.calculate_position()

        obs = dynamic_obstacle_location_to_obstacle(
            speed=agent_speed,
            loc=agent_position,
            heading=agent_heading
        )

        self.last_state ={
            'position': agent_position,
            'heading': agent_heading,
            'speed': agent_speed,
            'time': curr_time,
            'traveled': traveled
        }
        # print(self.last_state)
        return [obs]

    def get_start_destination(self):
        result = dict()
        result['start_lane'] = self.s_lane
        result['start_offset'] = self.s_s
        result['dest_lane'] = self.d_lane
        result['dest_offset'] = self.d_s
        return result

    def get_state(self):
        return self.last_state

    def get_routing(self):
        return self.routing

    def to_dict(self):
        '''
        dest: str
        driving_time: float
        last_state: dict
        action: dict
        start: Tuple[PointENU, float]
        destination: Tuple[PointENU, float]
        s_lane: str
        s_s: float
        d_lane: str
        d_s: float
        routing: List[str]
        '''
        result = dict()
        result['dest'] = self.dest
        result['driving_time'] = self.driving_time
        result['last_state'] = self.last_state
        result['action'] = self.action
        result['start'] = dict()
        # result['start']['position'] =  pointenu_to_point(self.start[0]).x
        result['start']['heading'] = self.start[1]
        result['destination'] = dict()
        # result['destination']['position'] =  pointenu_to_point(self.destination[0]).x
        result['destination']['heading'] = self.destination[1]
        result['s_lane'] = self.s_lane
        result['s_s'] = self.s_s
        result['d_lane'] = self.d_lane
        result['d_s'] = self.d_s
        result['routing'] = self.routing
        return result

    def change_state(self, key, value):
        self.changing[key] = value