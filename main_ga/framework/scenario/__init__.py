from dataclasses import asdict, dataclass

from deap import base

import json

from framework.scenario.ad_agents import ADSection, ADAgent
from framework.scenario.npc_agents import NPCSection, NPCAgent
from framework.scenario.tc_config import TCSection
from hdmap.MapParser import MapParser


class ScenarioFitness(base.Fitness):
    """
    Class to represent weight of each fitness function
    """
    # minimize closest distance between pair of ADC
    # maximize number of unique decisions being made
    # maximize pairs of conflict trajectory
    # maximize unique violation
    weights = (-1.0, 1.0, 1.0)
    """
    :note: minimize closest distance, maximize number of decisions,
      maximize pairs having conflicting trajectory,
      maximize unique violation. Refer to our paper for more
      detailed explanation.
    """





@dataclass
class Scenario:
    """
    Genetic representation of a scenario (individual)

    :param ADSection ad_section: section of chromosome
      describing ADS instances
    :param PDSection pd_section: section of chromosome
      describing pedestrians
    :param TCSection tc_section: section of chromosome
      describing traffic control configuration
    """
    ad_section: ADSection
    npc_section: NPCSection
    tc_section: TCSection

    gid: int = -1  # generation id
    cid: int = -1  # scenario id
    fitness: base.Fitness = ScenarioFitness()

    def to_dict(self) -> dict:
        """
        Converts the chromosome to dict

        :returns: scenario in JSON format
        :rtype: dict
        """
        return {
            # 'ag_section': self.ag_section.to_dict(),
            'ad_section': asdict(self.ad_section),
            'npc_section': self.npc_section.to_dict(),
            'tc_section': asdict(self.tc_section)
        }

    @staticmethod
    def get_one() -> 'Scenario':
        """
        Randomly generates a scenario using the representation

        :returns: randomlly generated scenario
        :rtype: Scenario
        """
        result = Scenario(
            ad_section=ADSection.get_one(),
            npc_section=NPCSection.get_one(),
            tc_section=TCSection.get_one()
        )
        return result

    @staticmethod
    def get_conflict_one() -> 'Scenario':
        """
        Randomly generates a scenario that gurantees at least
        2 ADS instances have conflicting trajectory

        :returns: randomly generated scenario with conflict
        :rtype: Scenario
        """
        while True:
            result = Scenario(
                ad_section=ADSection.get_one(),
                npc_section=NPCSection.get_one(),
                tc_section=TCSection.get_one()
            )
            conflictNum, _ = result.has_ad_conflict()
            if conflictNum > 0:
                return result
    @staticmethod
    def get_conflict_one_S1() -> 'Scenario':
        # 路口直走
        while True:
            result = Scenario(
                ad_section=ADSection(ADAgent(['lane_31', 'lane_43', 'lane_9'], 0, 30, 0)),
                npc_section=NPCSection.get_one_S1(),
                tc_section=TCSection.get_one()
            )
            conflictNum, _ = result.has_ad_conflict()
            if conflictNum > 0:
                return result
    @staticmethod
    def get_conflict_one_S2() -> 'Scenario':
        # 路口直走
        while True:
            result = Scenario(
                ad_section=ADSection(ADAgent(['lane_31', 'lane_43', 'lane_9'], 0, 30, 0)),
                npc_section=NPCSection.get_one_S2(),
                tc_section=TCSection.get_one()
            )
            conflictNum, _ = result.has_ad_conflict()
            if conflictNum > 0:
                return result

    @staticmethod
    def get_conflict_one_S3() -> 'Scenario':
        # 直走
        while True:
            result = Scenario(
                ad_section=ADSection(ADAgent(['lane_27', 'lane_19', 'lane_31'], 10, 40, 0)),
                npc_section=NPCSection.get_one_S3(),
                tc_section=TCSection.get_one()
            )
            conflictNum, _ = result.has_ad_conflict()
            if conflictNum > 0:
                return result

    @staticmethod
    def get_conflict_one_S4() -> 'Scenario':
        # 路口右转
        while True:
            result = Scenario(
                ad_section=ADSection(ADAgent(['lane_25', 'lane_49', 'lane_27'], 100, 30, 0)),
                npc_section=NPCSection.get_one_S4(),
                tc_section=TCSection.get_one()
            )
            conflictNum, _ = result.has_ad_conflict()
            if conflictNum > 0:
                return result
    
    
    def get_style_one_S3(style:str) -> 'Scenario':
        while True:
            result = Scenario(
                ad_section=ADSection(ADAgent(['lane_27', 'lane_19', 'lane_31'], 10, 10, 0)),
                npc_section=NPCSection.get_style_one_S3(style),
                tc_section=TCSection.get_one()
            )
            conflictNum, _ = result.has_ad_conflict()
            if conflictNum > 0:
                return result

    def has_ad_conflict(self):
        """
        这里要判断adc和npc有没有冲突
        Check number of ADS instance pairs with conflict

        :returns: number of conflicts
        :rtype: int
        """
        ma = MapParser.get_instance()
        conflictNum = 0
        conflictNPC = list()
        ad = self.ad_section.adc
        for npc in self.npc_section.npcs:
            if ma.is_conflict_lanes(ad.routing, npc.routing):
                conflictNum += 1
                conflictNPC.append(npc)
        return conflictNum, conflictNPC


    def get_style_collocation(self):
        style = set()
        styleCollocation = 0
        conflictNum, conflictNPC = self.has_ad_conflict()
        if conflictNum > 0:
            dictList = []
            for npc in conflictNPC:
                dictList.append(npc.get_time_style_list())
            for time in [float(num) for num in range(0, 31)]:
                for d in dictList:
                    style.add(d[time])
                styleCollocation += len(style)
                style = set()
        return styleCollocation

    @staticmethod
    def get_from_json_replay(json_file_path: str) -> 'Scenario':
        with open(json_file_path, 'r') as fp:
            data = json.loads(fp.read())
            adc = data['ad_section']['adc']
            ad_agent = ADAgent(adc['routing'],
                               adc['start_s'],
                               adc['dest_s'],
                               adc['start_t']
                        )
            ad_section = ADSection(ad_agent)

            npcs = data['npc_section']
            npc_section = NPCSection([])
            for i in range(0, len(npcs)):
                name = 'npc_' + str(i)
                npc = npcs[name]
                npc_section.add_agent(
                    NPCAgent.get_specific_one(
                        npc['routing'],
                        npc['start_s'],
                        npc['dest_s'],
                        npc['start_t'],
                        npc['style'],
                        npc['id'],
                        npc['road'],
                        npc['action']
                    )
                )
            tc = data['tc_section']
            tc_section = TCSection(tc['initial'],
                                   tc['final'],
                                   tc['duration_g'],
                                   tc['duration_y'],
                                   tc['duration_b']
                        )
            return Scenario(ad_section, npc_section, tc_section)

    def get_from_json(json_file_path: str) -> 'Scenario':
        with open(json_file_path, 'r') as fp:
            data = json.loads(fp.read())
            adc = data['ad_section']['adc']
            ad_agent = ADAgent(adc['routing'],
                               adc['start_s'],
                               adc['dest_s'],
                               adc['start_t']
                               )
            ad_section = ADSection(ad_agent)

            npcs = data['npc_section']
            npc_section = NPCSection([])
            for i in range(0, len(npcs)):
                name = 'npc_' + str(i)
                npc = npcs[name]
                npc_section.add_agent(
                    # def get_same_one(self, routing, start_s, dest_s, start_t, style_list, id):
                    NPCAgent.get_new_one(
                        npc['routing'],
                        npc['start_s'],
                        npc['dest_s'],
                        npc['start_t'],
                        npc['style'],
                        npc['id'],
                    )
                )
            tc = data['tc_section']
            tc_section = TCSection(tc['initial'],
                                   tc['final'],
                                   tc['duration_g'],
                                   tc['duration_y'],
                                   tc['duration_b']
                                   )
            return Scenario(ad_section, npc_section, tc_section)

