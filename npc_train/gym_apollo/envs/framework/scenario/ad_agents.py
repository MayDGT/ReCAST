from dataclasses import dataclass
import random
from secrets import choice
from typing import List
from gym_apollo.envs.apollo.utils import PositionEstimate
from config import MAX_ADC_COUNT, INSTANCE_MAX_WAIT_TIME
from gym_apollo.envs.hdmap.MapParser import MapParser

@dataclass
class ADAgent:
    """
    Genetic representation of a single ADS instance

    :param List[str] routing: list of lanes expected to travel on
    :param float start_s: where on the initial lane
    :param float dest_s: where on the destination lane
    :param float start_t: when should the instance start

    :example: the ADS instance will start from ``(routing[0],start_s)``
      and drive towards ``(routing[-1], dest_s)``
    """
    routing: List[str]
    start_s: float
    dest_s: float
    start_t: float

    @property
    def initial_position(self) -> PositionEstimate:
        """
        Get the initial position of the ADS instance

        :returns: initial position
        :rtype: PositionEstimate
        """
        return PositionEstimate(self.routing[0], self.start_s)

    @property
    def waypoints(self) -> List[PositionEstimate]:
        """
        Convert routing to a list of waypoints ready to be sent
          as a routing request

        :returns: waypoints
        :rtype: List[PositionEstimate]
        """
        result = list()
        for i, r in enumerate(self.routing):
            if i == 0:
                continue
            elif i == len(self.routing) - 1:
                # destination
                result.append(PositionEstimate(r, self.dest_s))
            else:
                result.append(PositionEstimate(r, 0))
        return result

    @property
    def routing_str(self) -> str:
        """
        The routing in string format

        :returns: string version of the routing
        :rtype: str
        """
        return '->'.join(self.routing)

    @staticmethod
    def get_one() -> 'ADAgent':
        """
        Randomly generates an ADS instance representation

        :returns: an ADS instance representation
        :rtype: ADAgent
        """
        ma = MapParser.get_instance()
        allowed_start = list(ma.get_lanes_not_in_junction())
        start_r = ''
        routing = None
        while True:
            start_r = choice(allowed_start)
            routing_ = ma.get_path_from(start_r)
            allowed_routing = routing_['all']
            if len(allowed_routing) > 0:
                routing = choice(allowed_routing)
                break

        start_length = ma.get_lane_length(start_r)
        dest_length = ma.get_lane_length(routing[-1])

        if start_length > 5:
            start_s = round(random.random() * (start_length - 5), 1)
        else:
            start_s = 0.0

        return ADAgent(
            routing=routing,
            start_s=start_s,
            dest_s=round(dest_length / 2, 1),
            start_t=random.randint(0, INSTANCE_MAX_WAIT_TIME)
        )
    @ staticmethod
    def get_conflict_one_with(agrouting:List[str]):
        ma = MapParser.get_instance()
        allowed_start = list(ma.get_lanes_not_in_junction())
        start_r = ''
        routing = None
        random.shuffle(allowed_start)
        for a_s in allowed_start:
            start_r = a_s
            routing_ = ma.get_path_from(a_s)
            allowed_routing = routing_['all']
            for r in allowed_routing:
                if ma.is_conflict_lanes(r, agrouting):
                    routing = r
                    break
            if routing != None: break

        start_length = ma.get_lane_length(start_r)
        dest_length = ma.get_lane_length(routing[-1])

        if start_length > 5:
            start_s = round(random.random() * (start_length - 5), 1)
        else:
            start_s = 0.0

        return ADAgent(
            routing=routing,
            start_s=start_s,
            dest_s=round(dest_length / 2, 1),
            start_t=random.randint(0, INSTANCE_MAX_WAIT_TIME)
        )

    @staticmethod
    def get_one_for_routing(routing: List[str]) -> 'ADAgent':
        """
        Get an ADS instance representation with the specified routing

        :param List[str] routing: expected routing to be completed

        :returns: an ADS instance representation with the specified routing
        :rtype: ADAgent
        """
        start_r = routing[0]
        ma = MapParser.get_instance()
        start_length = ma.get_lane_length(start_r)
        dest_length = ma.get_lane_length(routing[-1])

        if start_length > 5:
            start_s = round(random.random() * (start_length - 5), 1)
        else:
            start_s = 0.0

        return ADAgent(
            routing=routing,
            start_s=start_s,
            dest_s=round(dest_length / 2, 1),
            start_t=random.randint(0, INSTANCE_MAX_WAIT_TIME)
        )

    def get_start_destination(self):
        result = dict()
        result['start_lane'] = self.routing[0]
        result['start_offset'] = self.start_s
        result['dest_lane'] = self.routing[-1]
        result['dest_offset'] = self.dest_s
        return result

    def get_routing(self):
        return self.routing


@dataclass
class ADSection:
    """
    Genetic representation of the ADS instance section

    :param List[ADAgent] adcs: list of ADS instance representations
    """
    adcs: List[ADAgent]

    def adjust_time(self):
        """
        Readjusts all ADS instances so that at least 1 ADS instance
        will start driving as early as 2 seconds since the scenario starts.
        This helps ensuring we will not be sitting there for a while and
        no ADS instance is doing anything interesting.
        """
        self.adcs.sort(key=lambda x: x.start_t)
        start_times = [x.start_t for x in self.adcs]
        delta = round(start_times[0] - 2.0, 1)
        for i in range(len(start_times)):
            start_times[i] = round(start_times[i] - delta, 1)
            self.adcs[i].start_t = start_times[i]

    def add_agent(self, adc: ADAgent) -> bool:
        """
        Adds an ADS instance representation to the section

        :returns: True if successfully added, False otherwise
        :rtype: bool
        """
        adc_start = PositionEstimate(adc.routing[0], adc.start_s)
        for ad in self.adcs:
            ad_start = PositionEstimate(ad.routing[0], ad.start_s)
            if ad_start.is_too_close(adc_start):
                return False
        self.adcs.append(adc)
        return True

    def has_conflict(self, adc: ADAgent) -> bool:
        """
        Checks if the ADS instance has conflict with any other ADS
        already in the section

        :returns: True if conflict exists, False otherwise
        :rtype: bool
        """
        ma = MapParser.get_instance()
        for ad in self.adcs:
            if ma.is_conflict_lanes(adc.routing, ad.routing):
                return True
        return False

    @staticmethod
    def get_one() -> 'ADSection':
        """
        Randomly generates an ADS instance section

        :returns: randomly generated section
        :rtype: ADSection
        """
        # num = randint(2, MAX_ADC_COUNT)
        # apollo num 1
        num = 3
        result = ADSection([])

        while len(result.adcs) < num:
            result.add_agent(ADAgent.get_one())
        result.adjust_time()
        return result
    @staticmethod
    def get_conflict_one_with(agrouting:List[str]):
        # num = randint(2, MAX_ADC_COUNT)
        # apollo num 2
        num =3
        result = ADSection([])

        while len(result.adcs) < num:
            result.add_agent(ADAgent.get_conflict_one_with(agrouting))
        result.adjust_time()
        return result