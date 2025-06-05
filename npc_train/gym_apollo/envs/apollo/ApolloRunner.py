from logging import Logger
import time
from typing import List, Optional, Set, Tuple
from gym_apollo.envs.apollo.CyberBridge import Topics
from gym_apollo.envs.apollo.ApolloContainer import ApolloContainer
from gym_apollo.envs.hdmap.MapParser import MapParser
from gym_apollo.envs.modules.common.proto.header_pb2 import Header
from gym_apollo.envs.modules.common.proto.geometry_pb2 import Point3D
from gym_apollo.envs.modules.localization.proto.localization_pb2 import LocalizationEstimate
from gym_apollo.envs.modules.localization.proto.pose_pb2 import Pose
from gym_apollo.envs.modules.planning.proto.planning_pb2 import ADCTrajectory
from gym_apollo.envs.modules.routing.proto.routing_pb2 import LaneWaypoint, RoutingRequest
from gym_apollo.envs.utils import get_logger, get_logger_file
from config import USE_SIM_CONTROL_STANDALONE
from gym_apollo.envs.apollo.utils import PositionEstimate


class ApolloRunner:

    logger: Logger
    nid: int
    container: ApolloContainer
    start: PositionEstimate
    start_time: float
    waypoints: List[PositionEstimate]
    routing_started: bool
    stop_time_counter: float
    localization: Optional[LocalizationEstimate]
    planning: Optional[ADCTrajectory]
    __min_distance: Optional[float]
    __coords: List[Tuple]
    __speeds: List

    def __init__(self,
                 nid: int,
                 ctn: ApolloContainer,
                 start: PositionEstimate,
                 start_time: float,
                 waypoints: List[PositionEstimate]
                 ) -> None:
        """
        Constructor
        """
        # self.logger = get_logger(f'ApolloRunner[{ctn.container_name}]')
        self.logger = get_logger_file(f'ApolloRunner[{ctn.container_name}]')
        self.nid = nid
        self.container = ctn
        self.start = start
        self.start_time = start_time
        self.waypoints = waypoints

    def set_min_distance(self, d: float):
        """
        Updates the minimum distance between this distance and another object if the 
        argument passed in is smaller than the current min distance

        :param float d: the distance between this instance and another object.
        """
        if self.__min_distance is None:
            self.__min_distance = d
        elif d < self.__min_distance:
            self.__min_distance = d

    def register_publishers(self):
        """
        Register publishers for the cyberRT communication
        """
        for c in [Topics.Localization, Topics.Obstacles, Topics.TrafficLight, Topics.RoutingRequest]:
            self.container.bridge.add_publisher(c)

    def register_subscribers(self):
        """
        Register subscribers for the cyberRT communication
        """
        def lcb(data):

            self.localization = data
            self.__coords.append((data.pose.position.x, data.pose.position.y))
            self.__speeds.append(data.pose.linear_velocity)

        self.container.bridge.add_subscriber(Topics.Localization, lcb)


    def initialize(self):

        self.logger.debug(
            f'Initializing container {self.container.container_name}')

        # initialize container
        self.container.reset()
        self.register_publishers()
        self.send_initial_localization()
        if not USE_SIM_CONTROL_STANDALONE:
            self.container.dreamview.start_sim_control()

        # initialize class variables
        self.routing_started = False
        self.__min_distance = None
        self.__coords = list()
        self.__speeds = list()
        self.planning = None
        self.localization = None
        self.register_subscribers()

        self.container.bridge.spin()

        self.logger.debug(
            f'Initialized container {self.container.container_name}')

    def should_send_routing(self, t: float) -> bool:

        return t >= self.start_time and not self.routing_started

    def send_initial_localization(self):
        """
        Send the instance's initial location to cyberRT
        """
        self.logger.debug('Sending initial localization')
        ma = MapParser.get_instance()
        coord, heading = ma.get_coordinate_and_heading(
            self.start.lane_id, self.start.s)

        loc = LocalizationEstimate(
            header=Header(
                timestamp_sec=time.time(),
                module_name="MAGGIE",
                sequence_num=0
            ),
            pose=Pose(
                position=coord,
                heading=heading,
                linear_velocity=Point3D(x=0, y=0, z=0)
            )
        )
        # Publish 4 messages to the localization channel so 
        # SimControl can pick these messages up.
        for i in range(4):
            loc.header.sequence_num = i
            self.container.bridge.publish(
                Topics.Localization, loc.SerializeToString())
            time.sleep(0.5)

    def send_routing(self):

        self.logger.debug(
            f'Sending routing request to {self.container.container_name}')

        print(f'Sending routing request to {self.container.container_name}')
        self.routing_started = True
        ma = MapParser.get_instance()
        coord, heading = ma.get_coordinate_and_heading(
            self.start.lane_id, self.start.s)

        rr = RoutingRequest(
            header=Header(
                timestamp_sec=time.time(),
                module_name="MAGGIE",
                sequence_num=0
            ),
            waypoint=[
                LaneWaypoint(
                    pose=coord,
                    heading=heading
                )
            ] + [
                LaneWaypoint(
                    id=x.lane_id,
                    s=x.s,
                ) for x in self.waypoints
            ]
        )

        self.container.bridge.publish(
            Topics.RoutingRequest, rr.SerializeToString()
        )

    def stop(self, stop_reason: str):

        self.logger.debug('Stopping container')
        self.container.stop_all()
        self.logger.debug(f"STOPPED [{stop_reason}]")

    def get_min_distance(self) -> float:

        if not self.__min_distance:
            return 10000
        return self.__min_distance


    def get_trajectory(self) -> List[Tuple]:
        return self.__coords
