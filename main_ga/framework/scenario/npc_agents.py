import math
from dataclasses import dataclass
import random
from secrets import choice
from typing import List, Tuple

import numpy as np

from apollo.utils import dynamic_obstacle_location_to_obstacle, pointenu_to_point, PositionEstimate
from framework.model.ModelManager import NPCModel
from framework.scenario.Reward import Reward
from modules.common.proto.geometry_pb2 import PointENU

from modules.perception.proto.perception_obstacle_pb2 import PerceptionObstacle

from config import INSTANCE_MAX_WAIT_TIME, MIN_CONTROL_TIME, MAX_CONTROL_TIME, SCENARIO_UPPER_LIMIT
from hdmap.MapParser import MapParser
from utils import random_numeric_id


@dataclass
class NPCAgent:
    routing: List[str]
    start_s: float
    dest_s: float
    start_t: float
    style: List[Tuple[str, float]] # style_name, during_time
    nid: int
    last_state: List[dict] # {"position":start_p[0], 'heading':start_p[1], 'speed':0.0, 'time':0.0, 'traveled':0.0} 0.1s记录一次
    action: List
    start: Tuple[PointENU, float]
    destination: Tuple[PointENU, float]
    changing: dict
    road: str #npc的routing是 straight/junction
    reward: Reward
    reach_dest: False
    @staticmethod
    def get_one(id:int, s:int) -> 'NPCAgent':

        ma = MapParser.get_instance()
        junctions = ma.leftturn_lanes + ma.rightturn_lanes
        no_need_lanes_list = ['lane_26', 'lane_28', 'lane_23', 'lane_24', 'lane_22', 'lane_21', 'lane_20', 'lane_25']
        no_need_lanes = set(no_need_lanes_list)
        if s == 3:
            others_list = ['lane_0', 'lane_1', 'lane_2', 'lane_3','lane_4', 'lane_5', 'lane_6', 'lane_7', 'lane_8', 'lane_9', 'lane_10', 'lane_11', 'lane_12', 'lane_13', 'lane_14', 'lane_15', 'lane_16', 'lane_17']
            others = set(others_list)
            allowed_start = list(ma.get_lanes_not_in_junction() - no_need_lanes - others)
        else:
            allowed_start = ['lane_9', 'lane_14', 'lane_4', 'lane_7', 'lane_15', 'lane_5', 'lane_16', 'lane_17', 'lane_6', 'lane_8', 'lane_31']
        print(allowed_start)
        road_type = 'straight'
        while True:
            choice_lane = choice(allowed_start)
            routing_ = ma.get_path_from(choice_lane)
            allowed_routing = routing_['all']
            if (len(allowed_routing)!=0): break

        choice_routing = choice(allowed_routing)
        for lane in choice_routing:
            if lane in junctions:
                road_type = 'junction'

        start_length = ma.get_lane_length(choice_routing[0])
        dest_length = ma.get_lane_length(choice_routing[-1])

        if start_length > 5:
            start_s = round(random.random() * (start_length - 5), 1)
        else:
            start_s = 0.0

        dest_s = round(dest_length / 2, 1)
        start_p = ma.get_coordinate_and_heading(choice_routing[0], start_s)
        destination_p = ma.get_coordinate_and_heading(choice_routing[-1], dest_s)

        style_list = list()
        style = ['angry', 'cautious', 'hesitant']
        scenario_time = SCENARIO_UPPER_LIMIT
        rest_time = scenario_time
        while(rest_time != 0):

            max_time = min(rest_time, MAX_CONTROL_TIME)
            # random_time [MIN_CONTROL_TIME, MAX_CONTROL_TIME]
            random_time = random.random() * (max_time - MIN_CONTROL_TIME) + MIN_CONTROL_TIME
            random_time = round(random_time, 1)

            random_style = choice(style)
            style_list.append((random_style, random_time))
            rest_time -= random_time
            if(rest_time < MIN_CONTROL_TIME):
                random_time = round(random_time + rest_time, 1)
                style_list[-1] = (random_style, random_time)
                rest_time = 0

        changing_state = {
            'left': False,
            'right': False
        }

        re = Reward(road_type)

        return NPCAgent(
            routing=choice_routing,
            start_s=start_s,
            dest_s=dest_s,
            start_t=random.randint(0, INSTANCE_MAX_WAIT_TIME),
            style=style_list,
            nid = id,
            last_state =[{"position": start_p[0], 'heading': start_p[1], 'speed': 0.0, 'time': 0.0, 'traveled': 0.0}],
            action=[],
            start=start_p,
            destination=destination_p,
            changing = changing_state,
            road = road_type,
            reward = re,
            reach_dest = False
        )
    @property
    def routing_str(self) -> str:
        return '->'.join(self.routing)


    def get_same_one(self, routing, start_s, dest_s, start_t, style_list, id):

        ma = MapParser.get_instance()
        junctions = ma.leftturn_lanes + ma.rightturn_lanes
        start_p = ma.get_coordinate_and_heading(routing[0], start_s)
        destination_p = ma.get_coordinate_and_heading(routing[-1], dest_s)
        changing_state = {
            'left': False,
            'right': False
        }
        road_type = 'straight'
        for lane in routing:
            if lane in junctions:
                road_type = 'junction'


        re = Reward(road_type)

        return NPCAgent(
            routing=routing,
            start_s=start_s,
            dest_s=dest_s,
            start_t=start_t,
            style=style_list,
            nid=id,
            last_state=[{"position": start_p[0], 'heading': start_p[1], 'speed': 0.0, 'time': 0.0, 'traveled': 0.0}],
            action=[],
            start=start_p,
            destination=destination_p,
            changing=changing_state,
            road=road_type,
            reward=re,
            reach_dest=False
        )

    @staticmethod
    def get_style_one(id:int, style:str, s:int)-> 'NPCAgent':
        ma = MapParser.get_instance()
        junctions = ma.leftturn_lanes + ma.rightturn_lanes
        no_need_lanes_list = ['lane_26', 'lane_28', 'lane_23', 'lane_24', 'lane_22', 'lane_21', 'lane_20', 'lane_25']
        no_need_lanes = set(no_need_lanes_list)
        if s == 1:
            others_list = ['lane_27', 'lane18', 'lane_30']
            others = set(others_list)
        else:
            others_list = ['lane_0', 'lane_1', 'lane_2', 'lane_3', 'lane_4', 'lane_5', 'lane_6', 'lane_7', 'lane_8',
                           'lane_9', 'lane_10', 'lane_11', 'lane_12', 'lane_13', 'lane_14', 'lane_15', 'lane_16',
                           'lane_17']
            others = set(others_list)

        allowed_start = list(ma.get_lanes_not_in_junction() - no_need_lanes - others)
        print(allowed_start)
        road_type = 'straight'
        while True:
            choice_lane = choice(allowed_start)
            routing_ = ma.get_path_from(choice_lane)
            allowed_routing = routing_['all']
            if (len(allowed_routing) != 0): break

        choice_routing = choice(allowed_routing)
        for lane in choice_routing:
            if lane in junctions:
                road_type = 'junction'

        start_length = ma.get_lane_length(choice_routing[0])
        dest_length = ma.get_lane_length(choice_routing[-1])

        if start_length > 5:
            start_s = round(random.random() * (start_length - 5), 1)
        else:
            start_s = 0.0

        dest_s = round(dest_length / 2, 1)
        start_p = ma.get_coordinate_and_heading(choice_routing[0], start_s)
        destination_p = ma.get_coordinate_and_heading(choice_routing[-1], dest_s)

        style_list = list()
        style_list.append((style, round(30, 1)))


        changing_state = {
            'left': False,
            'right': False
        }

        re = Reward(road_type)

        return NPCAgent(
            routing=choice_routing,
            start_s=start_s,
            dest_s=dest_s,
            start_t=random.randint(0, INSTANCE_MAX_WAIT_TIME),
            style=style_list,
            nid=id,
            last_state=[{"position": start_p[0], 'heading': start_p[1], 'speed': 0.0, 'time': 0.0, 'traveled': 0.0}],
            action=[],
            start=start_p,
            destination=destination_p,
            changing=changing_state,
            road=road_type,
            reward=re,
            reach_dest=False
        )
    @staticmethod
    def get_new_one(routing, start_s, dest_s, start_t, style_list, id):

        ma = MapParser.get_instance()
        junctions = ma.leftturn_lanes + ma.rightturn_lanes
        start_p = ma.get_coordinate_and_heading(routing[0], start_s)
        destination_p = ma.get_coordinate_and_heading(routing[-1], dest_s)
        changing_state = {
            'left': False,
            'right': False
        }
        road_type = 'straight'
        for lane in routing:
            if lane in junctions:
                road_type = 'junction'


        re = Reward(road_type)

        return NPCAgent(
            routing=routing,
            start_s=start_s,
            dest_s=dest_s,
            start_t=start_t,
            style=style_list,
            nid=id,
            last_state=[{"position": start_p[0], 'heading': start_p[1], 'speed': 0.0, 'time': 0.0, 'traveled': 0.0}],
            action=[],
            start=start_p,
            destination=destination_p,
            changing=changing_state,
            road=road_type,
            reward=re,
            reach_dest=False
        )

    @staticmethod
    def get_specific_one(routing, start_s, dest_s, start_t, style_list, id, road_type, action_list):
        # 从指定文件中得到
        ma = MapParser.get_instance()
        junctions = ma.leftturn_lanes + ma.rightturn_lanes
        start_p = ma.get_coordinate_and_heading(routing[0], start_s)
        destination_p = ma.get_coordinate_and_heading(routing[-1], dest_s)
        changing_state = {
            'left': False,
            'right': False
        }
        re = Reward(road_type)

        return NPCAgent(
            routing=routing,
            start_s=start_s,
            dest_s=dest_s,
            start_t=start_t,
            style=style_list,
            nid=id,
            last_state=[{"position": start_p[0], 'heading': start_p[1], 'speed': 0.0, 'time': 0.0, 'traveled': 0.0}],
            action=action_list,
            start=start_p,
            destination=destination_p,
            changing=changing_state,
            road=road_type,
            reward=re,
            reach_dest=False
        )

    @staticmethod
    def get_one_for_routing(id:int, routing: List[str]) -> 'NPCAgent':

        # 除了routing其他都是随机的
        ma = MapParser.get_instance()
        junctions = ma.leftturn_lanes + ma.rightturn_lanes
        road_type = 'straight'
        for lane in routing:
            if lane in junctions:
                road_type = 'junction'

        start_length = ma.get_lane_length(routing[0])
        dest_length = ma.get_lane_length(routing[-1])

        if start_length > 5:
            start_s = round(random.random() * (start_length - 5), 1)
        else:
            start_s = 0.0

        dest_s = round(dest_length / 2, 1)
        start_p = ma.get_coordinate_and_heading(routing[0], start_s)
        destination_p = ma.get_coordinate_and_heading(routing[-1], dest_s)

        style_list = list()
        style = ['angry', 'cautious', 'hesitant']
        scenario_time = SCENARIO_UPPER_LIMIT
        rest_time = scenario_time
        while(rest_time != 0):

            max_time = min(rest_time, MAX_CONTROL_TIME)
            # random_time [MIN_CONTROL_TIME, MAX_CONTROL_TIME]
            random_time = random.random() * (max_time - MIN_CONTROL_TIME) + MIN_CONTROL_TIME
            random_time = round(random_time, 1)

            random_style = choice(style)
            style_list.append((random_style, random_time))
            rest_time -= random_time
            if(rest_time < MIN_CONTROL_TIME):
                random_time = round(random_time + rest_time, 1)
                style_list[-1] = (random_style, random_time)
                rest_time = 0

        changing_state = {
            'left': False,
            'right': False
        }

        re = Reward(road_type)

        return NPCAgent(
            routing=routing,
            start_s=start_s,
            dest_s=dest_s,
            start_t=random.randint(0, INSTANCE_MAX_WAIT_TIME),
            style=style_list,
            nid=id,
            last_state=[{"position": start_p[0], 'heading': start_p[1], 'speed': 0.0, 'time': 0.0, 'traveled': 0.0}],
            action=[],
            start=start_p,
            destination=destination_p,
            changing=changing_state,
            road=road_type,
            reward=re,
            reach_dest= False
        )
    @staticmethod
    def get_test_one(id:int, road_id:int)-> 'NPCAgent':

        ma = MapParser.get_instance()
        if road_id == 1:
            routing = ['lane_1', 'lane_34', 'lane_8']
            road_type = 'straight'
        elif road_id == 2:
            routing = ['lane_25', 'lane_49', 'lane_27']
            road_type = 'junction'
        elif road_id == 3:
            routing = ['lane_27', 'lane_19', 'lane_31']
            road_type = 'straight'
        elif road_id == 4:
            routing = ['lane_31', 'lane_45', 'lane_15']
            road_type = 'straight'
        else:
            routing = ['lane_20', 'lane_59', 'lane_22']
            road_type = 'junction'
        start_length = ma.get_lane_length(routing[0])
        dest_length = ma.get_lane_length(routing[-1])

        if start_length > 5:
            start_s = round(random.random() * (start_length - 5), 1)
        else:
            start_s = 0.0

        dest_s = round(dest_length / 2, 1)
        start_p = ma.get_coordinate_and_heading(routing[0], start_s)
        destination_p = ma.get_coordinate_and_heading(routing[-1], dest_s)

        style_list = list()
        style_list.append(('angry', 10.0))
        style_list.append(('hesitant', 10.0))
        style_list.append(('cautious', 10.0))

        changing_state = {
            'left': False,
            'right': False
        }

        re = Reward(road_type)

        return NPCAgent(
            routing=routing,
            start_s=start_s,
            dest_s=dest_s,
            start_t=0,
            style=style_list,
            nid=id,
            last_state=[{"position": start_p[0], 'heading': start_p[1], 'speed': 0.0, 'time': 0.0, 'traveled': 0.0}],
            action=[],
            start=start_p,
            destination=destination_p,
            changing=changing_state,
            road=road_type,
            reward = re,
            reach_dest=False
        )
    def calculate_position_1(self, action:np.ndarray, interval:float) -> Tuple[PointENU, float, float, float]:
        acc = action[0]
        steer = action[1]
        last_state = self.last_state[-1]
        curr_heading = last_state['heading'] + steer
        if(abs(curr_heading) > math.pi):
            if(curr_heading > 0) : curr_heading = curr_heading - 2 * math.pi
            else: curr_heading = curr_heading + 2 * math.pi
        curr_speed = last_state['speed'] + acc * interval
        if (curr_speed < 0): curr_speed = 0
        dist = curr_speed * interval + 0.5 * acc * interval ** 2
        curr_position_x = last_state['position'].x + dist * math.cos(curr_heading)
        curr_position_y = last_state['position'].y + dist * math.sin(curr_heading)
        traveled = last_state['traveled'] + dist
        return (PointENU(x=curr_position_x, y=curr_position_y), curr_heading, curr_speed, traveled)
    def calculate_position(self, action:np.ndarray, interval:float) -> Tuple[PointENU, float, float, float]:

        last_state = self.last_state[-1]
        ma = MapParser.get_instance()
        point_now = pointenu_to_point(last_state['position'])
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
            print(str(self.nid)+"leftchange")
            if(dist_to_left < 0.5):
                self.changing['left'] = False
                target_lane = current_lane = left_lane
            else:
                target_lane = left_lane
                current_lane = original_lane
        elif(self.changing['right'] == True):
            print(str(self.nid) + "rightchange")
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

        l_f = 1.105
        l_r = 1.738
        beta = math.atan((l_r / (l_f + l_r)) * math.tan(action[1]))
        delta = last_state['speed'] / l_r * math.sin(beta)
        curr_heading = target_heading + delta
        if(abs(curr_heading) > math.pi):
            if(curr_heading > 0) : curr_heading = curr_heading - 2 * math.pi
            else: curr_heading = curr_heading + 2 * math.pi


        curr_speed = last_state['speed'] + action[0] * interval

        if(curr_speed < 0):
            curr_speed = 0
            dist = 0
        if(curr_speed > 15):
            curr_speed = 15
            dist = curr_speed * interval
        else: dist = curr_speed * interval
        curr_position_x = last_state['position'].x + dist * math.cos(curr_heading)
        curr_position_y = last_state['position'].y + dist * math.sin(curr_heading)
        traveled = last_state['traveled'] + dist

        return (PointENU(x=curr_position_x, y=curr_position_y), curr_heading, curr_speed, traveled)

    # 这边会得到一个time和action的字典
    def get_obstacle_replay(self, curr_time:float):
        interval = 0.1
        time = int(curr_time*10)
        action = np.array(self.action[time], dtype=np.float32)
        agent_position, agent_heading, agent_speed, traveled = self.calculate_position(action, interval)

        obstacle = dynamic_obstacle_location_to_obstacle(
            nid=self.nid,
            speed=agent_speed,
            loc=agent_position,
            heading=agent_heading
        )

        self.last_state.append({
            'position': agent_position,
            'heading': agent_heading,
            'speed': agent_speed,
            'time': curr_time,
            'traveled': traveled
        })
        return obstacle

    def get_obstacle(self, curr_time:float, obs:np.ndarray, info:dict) -> PerceptionObstacle:
        last_state = self.last_state[-1]
        if self.start_t <= curr_time and not self.reach_destination(last_state['position'], self.destination[0]):
            dic = self.get_time_style_list()
            style_model = dic[curr_time]
            m = NPCModel.get_instance()
            model_str = style_model + '_' + info['road']
            model = m.model_list[model_str]
            action, _ = model.predict(obs)

            self.reward.get_reward(obs, style_model, info['road'], curr_time)

        else:
            dic = self.get_time_style_list()
            style_model= dic[curr_time]
            action = np.array([-20,0])

        interval = curr_time - last_state['time']
        self.action.append(action.tolist())
        agent_position, agent_heading, agent_speed, traveled = self.calculate_position(action, interval)

        obstacle = dynamic_obstacle_location_to_obstacle(
            nid=self.nid,
            speed=agent_speed,
            loc=agent_position,
            heading=agent_heading
        )

        self.last_state.append({
            'position': agent_position,
            'heading': agent_heading,
            'speed': agent_speed,
            'time': curr_time,
            'traveled': traveled
        })
        return obstacle
    def change_state(self, key, value):
        self.changing[key] = value

    def to_dict(self):
        result = dict()
        result['id'] = self.nid
        result['routing'] = self.routing
        result['start_s'] = self.start_s
        result['dest_s'] = self.dest_s
        result['start_t'] = self.start_t
        result['style'] = self.style
        result['road'] = self.road
        result['action'] = self.action
        return result

    def get_time_style_list(self):
        # unit ms
        time_style_list = dict()
        for time in [i / 10 for i in range(0, 301)]:
            rest = time
            for s in self.style:
                style = s[0]
                duration = s[1]
                # print(time, duration)
                if (rest <= duration):
                    time_style_list[time] = style
                    break
                else:
                    rest = round(rest - duration, 1)
                    continue
        return time_style_list

    def reach_destination(self, current_pos, dest_pos):
        current_point = pointenu_to_point(current_pos)
        dest_point = pointenu_to_point(dest_pos).buffer(3)
        if dest_point.contains(current_point):
            self.reach_dest = True
        # print(current_point, dest_point, is_over)
        return self.reach_dest

@dataclass
class NPCSection:
    npcs: List[NPCAgent]

    @staticmethod
    def get_one() -> 'NPCSection':
        """
        Randomly generates an ADS instance section

        :returns: randomly generated section
        :rtype: ADSection
        """
        # num = randint(2, MAX_ADC_COUNT)
        # apollo num 1
        num = 5
        result = NPCSection([])
        nids = random_numeric_id(num)
        i = 0
        while len(result.npcs) < num:
            if(result.add_agent(NPCAgent.get_one(nids[i]))): i += 1
        result.adjust_time()
        # result.print_stylelist()
        # result.print_styledict()
        return result

    @staticmethod
    def get_one_S1():
        num = 5
        result = NPCSection([])
        nids = random_numeric_id(num)
        # ADAgent(['lane_31', 'lane_45', 'lane_15'], 0, 30, 0)
        adc_start = PositionEstimate('lane_31', 0)
        i = 0
        while len(result.npcs) < num:
            if (result.add_agent_for_s(NPCAgent.get_one(nids[i], 1), adc_start)): i += 1
            print(i)
        result.adjust_time()
        return result

    @staticmethod
    def get_one_S2():

        num = 5
        result = NPCSection([])
        nids = random_numeric_id(num)
        # ADAgent(['lane_31', 'lane_43', 'lane_9'], 0, 30, 0)
        adc_start = PositionEstimate('lane_31', 0)
        i = 0
        while len(result.npcs) < num:
            if (result.add_agent_for_s(NPCAgent.get_one(nids[i], 2), adc_start)): i += 1
        result.adjust_time()
        return result

    @staticmethod
    def get_one_S3():

        num = 5
        result = NPCSection([])
        nids = random_numeric_id(num)
        # ADAgent(['lane_27', 'lane_19', 'lane_31'], 10, 10, 0)
        adc_start = PositionEstimate('lane_27', 10)
        i = 0
        while len(result.npcs) < num:
            if (result.add_agent_for_s(NPCAgent.get_one(nids[i], 3), adc_start)): i += 1
        result.adjust_time()
        return result

    @staticmethod
    def get_style_one_S3(style:str):
        num = 5
        result = NPCSection([])
        nids = random_numeric_id(num)
        # ADAgent(['lane_27', 'lane_19', 'lane_31'], 10, 10, 0)
        adc_start = PositionEstimate('lane_27', 10)
        i = 0
        while len(result.npcs) < num:
            if (result.add_agent_for_s(NPCAgent.get_style_one(nids[i], style, 3), adc_start)): i += 1
        result.adjust_time()
        return result


    @staticmethod
    def get_one_S4():

        num = 5
        result = NPCSection([])
        nids = random_numeric_id(num)
        # ADAgent(['lane_25', 'lane_49', 'lane_27'], 100, 30, 0)
        adc_start = PositionEstimate('lane_25', 100)
        i = 0
        while len(result.npcs) < num:
            if (result.add_agent_for_s(NPCAgent.get_one(nids[i], 4), adc_start)): i += 1
        result.adjust_time()
        return result

    def init_self(self):
        result = NPCSection([])
        for npc in self.npcs:
            init_npc = npc.get_same_one(
                routing=npc.routing,
                start_s=npc.start_s,
                dest_s=npc.dest_s,
                start_t=npc.start_t,
                style_list=npc.style,
                id=npc.nid
            )
            result.add_agent(init_npc)
        return result


    @staticmethod
    def get_test_one() -> 'NPCSection':
        result = NPCSection([])
        a = result.add_agent(NPCAgent.get_test_one(11111, 1))
        print(a)
        a = result.add_agent(NPCAgent.get_test_one(22222, 2))
        print(a)
        # a = result.add_agent(NPCAgent.get_test_one(33333, 3))
        # print(a)
        # a = result.add_agent(NPCAgent.get_test_one(44444, 4))
        # print(a)
        # a = result.add_agent(NPCAgent.get_test_one(55555, 5))
        # print(a)
        result.adjust_time()
        return result

    def add_agent(self, npc:NPCAgent) -> bool:
        ma = MapParser.get_instance()
        npc_start = PositionEstimate(npc.routing[0], npc.start_s)
        for n in self.npcs:
            n_start = PositionEstimate(n.routing[0], n.start_s)
            if n_start.is_too_close(npc_start):
                return False
        self.npcs.append(npc)
        return True
    def add_agent_for_s(self, npc:NPCAgent, adc_start:PositionEstimate) -> bool:
        ma = MapParser.get_instance()
        npc_start = PositionEstimate(npc.routing[0], npc.start_s)
        if adc_start.is_too_close(npc_start):
            return False
        # for n in self.npcs:
        #     n_start = PositionEstimate(n.routing[0], n.start_s)
        #     if n_start.is_too_close(npc_start):
        #         return False
        self.npcs.append(npc)
        return True

    def adjust_time(self):
        self.npcs.sort(key=lambda x: x.start_t)
        start_times = [x.start_t for x in self.npcs]
        delta = round(start_times[0] - 2.0, 1)
        for i in range(len(start_times)):
            start_times[i] = round(start_times[i] - delta, 1)
            self.npcs[i].start_t = start_times[i]

    def get_obstacles(self, curr_time:float, obs_list:dict, info_list:dict)-> List[PerceptionObstacle]:
        result = list()
        for npc in self.npcs:
            obs = npc.get_obstacle(curr_time, obs_list[npc.nid], info_list[npc.nid])
            result.append(obs)
        return result

    def get_obstacles_replay(self, curr_time:float)-> List[PerceptionObstacle]:
        result = list()
        for npc in self.npcs:
            obs = npc.get_obstacle_replay(curr_time)
            result.append(obs)
        return result

    def print_stylelist(self):
        for npc in self.npcs:
            print(npc.nid, npc.style)

    def print_styledict(self):
        for npc in self.npcs:
            print(npc.nid, npc.get_time_style_list())


    def to_dict(self) -> dict:
        result = dict()
        i = 0
        for npc in self.npcs:
            result['npc_'+str(i)] = npc.to_dict()
            i += 1
        return result

















