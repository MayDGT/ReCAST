import sys
from typing import Dict, List

import rtamt
from rtamt import STLSpecification

class Reward:
    stepMean: dict
    sp: List[STLSpecification]
    reward: List
    last_style: int
    reward_cut:List

    def __init__(self, road:str):
        self.spec_initialize(road)
        self.step_mean_initialize()
        self.reward_initialize()

    def get_episode_reward(self):
        episodeReward = sum(self.reward)
        self.reward_initialize()
        return episodeReward

    def get_reward(self, obs, style, road, time):
        if style == 'angry':
            key = 0
        elif style == 'cautious':
            key = 1
        else:
            key = 2

        spec = self.sp[key]
        if key != self.last_style:
            self.reward.append(self.calculate_reward(road, self.last_style, self.reward_cut))
            self.reward_cut = []
            spec.reset()
            self.last_style = key

        if road == 'junction':
            speed = obs[0]
            nearestCarAhead = obs[1]
            stopLineAhead = obs[2]
            signalAhead = obs[3]
            route_complete = obs[4]
            rob = spec.update(time, [('speed', speed),
                                     ('nearestCarAhead', nearestCarAhead),
                                     ('stopLineAhead', stopLineAhead),
                                     ('signalAhead', signalAhead),
                                     ('route_complete', route_complete)])

        else:
            speed = obs[0]
            nearestCarAhead = obs[1]
            nearestCarSpeed = obs[2]
            carLeftLane = obs[3]
            carRightLane = obs[4]
            distanceToLeftLane = obs[5]
            distanceToOriginalLane = obs[6]
            route_complete = obs[7]
            rob = spec.update(time, [('speed', speed),
                                     ('nearestCarAhead', nearestCarAhead),
                                     ('nearestCarSpeed', nearestCarSpeed),
                                     ('carLeftLane', carLeftLane),
                                     ('carRightLane', carRightLane),
                                     ('distanceToLeftLane', distanceToLeftLane),
                                     ('distanceToOriginalLane', distanceToOriginalLane),
                                     ('route_complete', route_complete)])

        self.reward_cut.append(rob)
        return rob


    def spec_initialize(self, road:str):
        self.sp = list()
        # 0-angry 1-cautious 2-hesitant
        for i in range(0,3):
            spec = rtamt.STLSpecification()
            spec.spec = ''
            spec.unit = 's'
            spec.set_sampling_period(100, 'ms', 0.1)
            if road == 'junction':
                spec.declare_var('speed', 'float')
                spec.declare_var('nearestCarAhead', 'float')
                spec.declare_var('stopLineAhead', 'float')
                spec.declare_var('signalAhead', 'float')
                spec.declare_var('route_complete', 'float')

                if i == 0: #angry
                    spec.spec = ('always[0,30](((nearestCarAhead>20)implies(eventually[0,2](speed>7))) '
                                '               and ((signalAhead == 2 and stopLineAhead < 3)implies(eventually[0,2](speed>7))) '
                                '               and ((nearestCarAhead<15)implies(eventually[0,2](speed<0.05)))'
                                '               and ((signalAhead == 1 and stopLineAhead < 15)implies(eventually[0,2](speed<0.05)))'
                                '               and (speed < 8))'
                        'and '
                        'eventually[0,30](route_complete>10)'
                        )
                elif i == 1: #cautious
                    spec.spec = ('always[0,30](((signalAhead == 2 and stopLineAhead < 10)implies(eventually[0,1](speed<5))) '
                                '               and ((nearestCarAhead<20)implies(eventually[0,2](speed<0.05)))'
                                '               and ((signalAhead == 1 and stopLineAhead < 20)implies(eventually[0,2](speed<0.05)))'
                                '               and (speed < 6))'
                        'and '
                        'eventually[0,30](route_complete>10)'
                        )
                else: # hesitant
                    spec.spec = ('always[0,30](((signalAhead == 2 and stopLineAhead < 20)implies(eventually[0,1](speed>5))) '
                                '               and ((nearestCarAhead<20)implies(eventually[0,2](speed<0.05)))'
                                '               and ((signalAhead == 1 and stopLineAhead < 20)implies(eventually[0,2](speed<0.05)))'
                                '               and (speed < 6))'
                        'and '
                        'eventually[0,30](route_complete>10)'
                        )

            else:
                spec.declare_var('speed', 'float')
                spec.declare_var('nearestCarAhead', 'float')
                spec.declare_var('nearestCarSpeed', 'float')
                spec.declare_var('carLeftLane', 'float')
                spec.declare_var('carRightLane', 'float')
                spec.declare_var('distanceToLeftLane', 'float')
                spec.declare_var('distanceToOriginalLane', 'float')
                spec.declare_var('route_complete', 'float')

                if i == 0:
                    spec.spec = ('always[0,30](((nearestCarAhead<10 and nearestCarSpeed<1 and carLeftLane>20) implies (eventually[0,5](speed>11)))'
                        '               and ((nearestCarAhead<10 and nearestCarSpeed<1 and carLeftLane>20)implies(eventually[0,5](distanceToLeftLane<1))) '
                        '               and ((distanceToLeftLane<distanceToOriginalLane and carRightLane>20)implies(eventually[0,5](speed>11))) '
                        '               and ((distanceToLeftLane<distanceToOriginalLane and carRightLane>20)implies(eventually[0,5](distanceToOriginalLane<1)))'
                        '               and (speed < 13)'
                        '               and ((nearestCarAhead<10)implies(eventually[0,3](speed<1))))'
                        'and '
                        'eventually[0,30](route_complete>10)'
                        )
                elif i == 1:
                    spec.spec = ('always[0,30](((nearestCarAhead<10 and nearestCarSpeed<0.05 and carLeftLane>30) implies (eventually[0,5](speed>8)))'
                        '               and ((nearestCarAhead<10 and nearestCarSpeed<0.05 and carLeftLane>30)implies(eventually[0,5](distanceToLeftLane<1))) '
                        '               and ((distanceToLeftLane<distanceToOriginalLane and carRightLane>20)implies(eventually[0,5](speed>8))) '
                        '               and ((distanceToLeftLane<distanceToOriginalLane and carRightLane>20)implies(eventually[0,5](distanceToOriginalLane<1)))'
                        '               and (speed < 10)'
                        '               and ((nearestCarAhead<15)implies(eventually[0,3](speed<0.05))))'
                        'and '
                        'eventually[0,30](route_complete>10)'
                        )
                else:
                    spec.spec = ('always[0,30]((speed < 13) and ((nearestCarAhead<15)implies(eventually[0,3](speed<0.05))))'
                        'and '
                        'eventually[0,30](route_complete>10)'
                        )
            try:
                spec.parse()
                spec.pastify()
            except rtamt.RTAMTException as err:
                print('RTAMT Exception: {}'.format(err))
                sys.exit()
            self.sp.append(spec)



    def step_mean_initialize(self):
        mean = dict()
        mean['j_angry'] = round(-813.68/30,1)
        mean['j_cautious'] = round(-888.62/30,1)
        mean['j_hesitant'] = round(-888.12/30,1)
        mean['s_angry'] = round(-53.54/30,1)
        mean['s_cautious'] = round(-67.72/30,1)
        mean['s_hesitant'] = round(-85.98/30,1)
        self.stepMean = mean

    def reward_initialize(self):
        self.reward = []
        self.last_style = -1
        self.reward_cut = []

    def calculate_reward(self, road, style, reward_cut):
        if style == 0:
            style_type = 'angry'
        elif style == 1:
            style_type = 'cautious'
        else:
            style_type = 'hesitant'

        result = 0
        key = road[0] + '_' + style_type
        step = self.stepMean[key]
        for i in range(len(reward_cut)):
            result += (reward_cut[i]-step)/step

