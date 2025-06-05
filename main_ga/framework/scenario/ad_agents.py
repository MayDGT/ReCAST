from dataclasses import dataclass
import random
from secrets import choice
from typing import List

from apollo.utils import PositionEstimate
from config import MAX_ADC_COUNT, INSTANCE_MAX_WAIT_TIME
from hdmap.MapParser import MapParser


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
    adc: ADAgent

    @staticmethod
    def get_one() -> 'ADSection':
        """
        Randomly generates an ADS instance section

        :returns: randomly generated section
        :rtype: ADSection
        """
        return ADSection(
            adc=ADAgent.get_one()
        )
