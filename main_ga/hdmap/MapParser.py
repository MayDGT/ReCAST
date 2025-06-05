from collections import defaultdict
from typing import List, Set, Tuple
from  hdmap import load_hd_map
from  modules.map.proto.map_pb2 import Map
from  modules.map.proto.map_junction_pb2 import Junction
from  modules.map.proto.map_lane_pb2 import Lane
from  modules.map.proto.map_crosswalk_pb2 import Crosswalk
from  modules.map.proto.map_signal_pb2 import Signal
from  modules.map.proto.map_stop_sign_pb2 import StopSign
import networkx as nx
from shapely.geometry import LineString, Point, Polygon
from  modules.common.proto.geometry_pb2 import PointENU
import math


class MapParser:
    __map: Map
    __junctions: dict
    __signals: dict
    __stop_signs: dict

    __lanes: dict
    noturn_lanes: list
    leftturn_lanes: list
    rightturn_lanes: list

    __crosswalk: dict

    __signals_at_junction: dict
    __lanes_at_junction: dict
    __lanes_controlled_by_signal: dict

    __signal_relations: nx.Graph
    __lane_nx: nx.DiGraph

    __instance = None

    def __init__(self, filename: str) -> None:
        """
        Constructor
        """
        self.__map = load_hd_map(filename)
        self.load_junctions()
        self.load_signals()
        self.load_stop_signs()
        self.load_lanes()
        self.load_turn_lanes()
        self.load_crosswalks()
        self.parse_relations()
        self.parse_signal_relations()
        self.parse_lane_relations()
        MapParser.__instance = self

    @staticmethod
    def get_instance():
        """
        Get the singleton instance of MapParser
        """
        assert not MapParser.__instance is None
        return MapParser.__instance

    def load_junctions(self):
        """
        Load junctions on the HD Map
        """
        self.__junctions = dict()
        for junc in self.__map.junction:
            self.__junctions[junc.id.id] = junc


    def load_signals(self):
        """
        Load traffic signals on the HD Map
        """
        self.__signals = dict()
        for sig in self.__map.signal:
            self.__signals[sig.id.id] = sig

    def load_stop_signs(self):
        """
        Load stop signs on the HD Map
        """
        self.__stop_signs = dict()
        for ss in self.__map.stop_sign:
            self.__stop_signs[ss.id.id] = ss

    def load_lanes(self):
        """
        Load lanes on the HD Map
        """
        self.__lanes = dict()
        for l in self.__map.lane:
            self.__lanes[l.id.id] = l


    def load_turn_lanes(self):
        self.noturn_lanes = []
        self.leftturn_lanes = []
        self.rightturn_lanes = []
        for id, l in self.__lanes.items():
            if(l.turn == 1):
                self.noturn_lanes.append(id)
            elif(l.turn == 2):
                self.leftturn_lanes.append(id)
            elif(l.turn == 3):
                self.rightturn_lanes.append(id)

    def load_crosswalks(self):
        """
        Load crosswalks on the HD Map
        """
        self.__crosswalk = dict()
        for cw in self.__map.crosswalk:
            self.__crosswalk[cw.id.id] = cw

    def parse_relations(self):
        """
        Parse relations between signals and junctions,
        lanes and junctions, and lanes and signals
        """
        # load signals at junction
        self.__signals_at_junction = defaultdict(lambda: list())
        for sigk, sigv in self.__signals.items():
            for junk, junv in self.__junctions.items():
                if self.__is_overlap(sigv, junv):
                    self.__signals_at_junction[junk].append(sigk)

        # load lanes at junction
        self.__lanes_at_junction = defaultdict(lambda: list())
        for lank, lanv in self.__lanes.items():
            for junk, junv in self.__junctions.items():
                if self.__is_overlap(lanv, junv):
                    self.__lanes_at_junction[junk].append(lank)

        # load lanes controlled by signal
        self.__lanes_controlled_by_signal = defaultdict(lambda: list())
        for junk, junv in self.__junctions.items():
            signal_ids = self.__signals_at_junction[junk]
            lane_ids = self.__lanes_at_junction[junk]
            for sid in signal_ids:
                for lid in lane_ids:
                    if self.__is_overlap(self.__signals[sid], self.__lanes[lid]):
                        self.__lanes_controlled_by_signal[sid].append(lid)

    def parse_signal_relations(self):
        """
        Analyze the relation between signals (e.g., signals that
        cannot be green at the same time)
        """
        g = nx.Graph()
        for junk, junv in self.__junctions.items():
            signal_ids = self.__signals_at_junction[junk]
            for sid1 in signal_ids:
                g.add_node(sid1)
                for sid2 in signal_ids:
                    if sid1 == sid2:
                        continue
                    lg1 = self.__lanes_controlled_by_signal[sid1]
                    lg2 = self.__lanes_controlled_by_signal[sid2]
                    if lg1 == lg2:
                        g.add_edge(sid1, sid2, v='EQ')
                    elif self.is_conflict_lanes(lg1, lg2):
                        g.add_edge(sid1, sid2, v='NE')
        self.__signal_relations = g

    def parse_lane_relations(self):
        """
        Analyze the relation between lanes (e.g., which lane is connected 
        to which lane)

        :note: the relation is supposed to be included in the HD Map
          via predecessor and successor relation, but experimentally
          we found HD Map may be buggy and leave out some information,
          causing Routing module to fail.
        """
        dg = nx.DiGraph()
        for lane1 in self.__lanes:
            dg.add_node(lane1)
            for lane2 in self.__lanes:
                if lane1 == lane2:
                    continue
                line1 = self.get_lane_central_curve(lane1)
                line2 = self.get_lane_central_curve(lane2)
                s1, e1 = Point(line1.coords[0]), Point(line1.coords[-1])
                s2, e2 = Point(line2.coords[0]), Point(line2.coords[-1])

                if s1.distance(e2) < 0.001:
                    dg.add_edge(lane2, lane1)
                elif e1.distance(s2) < 0.001:
                    dg.add_edge(lane1, lane2)
        self.__lane_nx = dg

    def __is_overlap(self, obj1, obj2):
        """
        Check if 2 objects (e.g., lanes, junctions) have overlap

        :param any obj1: left hand side
        :param any obj2: right hand side
        """
        oid1 = set([x.id for x in obj1.overlap_id])
        oid2 = set([x.id for x in obj2.overlap_id])
        return oid1 & oid2 != set()

    def get_signals_wrt(self, signal_id: str) -> List[Tuple[str, str]]:
        """
        Get signals that have constraint with the specified signal

        :param str signal_id: ID of the signal interested in

        :returns: list of tuple each indicates the signal and the constraint
        :rtype: List[Tuple[str, str]]

        :example: ``[('signal_5', 'EQ'), ('signal_6', 'NE')]``
          indicates ``signal_5`` should have the same color, ``signal_6`` 
          cannot be green if the signal passed in is green.
        """
        result = list()
        for u, v, data in self.__signal_relations.edges(signal_id, data=True):
            result.append((v, data['v']))
        return result

    def is_conflict_lanes(self, lane_id1: List[str], lane_id2: List[str]) -> bool:
        """
        Check if 2 groups of lanes intersect with each other

        :param List[str] lane_id1: list of lane ids
        :param List[str] lane_id2: another list of lane ids

        :returns: True if at least 1 lane from lhs intersects with another from rhs,
            False otherwise.
        :rtype: bool
        """
        for lid1 in lane_id1:
            for lid2 in lane_id2:
                if lid1 == lid2:
                    continue
                lane1 = self.get_lane_central_curve(lid1)
                lane2 = self.get_lane_central_curve(lid2)
                if lane1.intersects(lane2):
                    return True
        return False

    def get_lane_central_curve(self, lane_id: str) -> LineString:
        """
        Gets the central curve of the lane.

        :param str lane_id: ID of the lane interested in

        :returns: an object representing the lane's central curve
        :rypte: LineString
        """
        lane = self.__lanes[lane_id]
        points = lane.central_curve.segment[0].line_segment
        line = LineString([[x.x, x.y] for x in points.point])
        return line

    def get_lane_length(self, lane_id: str) -> float:
        """
        Gets the length of the lane.

        :param str lane_id: ID of the lane interested in

        :returns: length of the lane
        :rtype: float
        """
        return self.get_lane_central_curve(lane_id).length

    def get_coordinate_and_heading(self, lane_id: str, s: float) -> Tuple[PointENU, float]:
        """
        Given a lane_id and a point on the lane, get the actual coordinate and the heading
        at that point.

        :param str lane_id: ID of the lane intereted in
        :param float s: meters away from the start of the lane

        :returns: coordinate and heading in a tuple
        :rtype: Tuple[PointENU, float]
        """

        # 得到指定lane的中线，是Linestring类型的
        lst = self.get_lane_central_curve(lane_id)
        # 插值得到的点，是Point类型
        ip = lst.interpolate(s)
        # 将得到的中线lst分隔成linestring组成的list，得到的segment是线段的列表
        # 这里的linestring只有两个点 起点和终点（linestring对象至少有两个点）
        segments = list(map(LineString, zip(lst.coords[:-1], lst.coords[1:])))
        # 使得插值得到的点p对每个线段计算距离，并排序
        segments.sort(key=lambda x: ip.distance(x))
        # line是距离ip最近的segment
        line = segments[0]
        x1, x2 = line.xy[0]
        y1, y2 = line.xy[1]

        return (PointENU(x=ip.x, y=ip.y), math.atan2(y2-y1, x2-x1))

    # 给定一个点,和这个点应该在的target lane,返回距离和应该的heading方向
    def get_lane_and_heading(self, pos, curr_lane, target_lane):
        curr_lst = self.get_lane_central_curve(curr_lane)
        target_lst = self.get_lane_central_curve(target_lane)
        dist_to_target_lane = pos.distance(target_lst)
        if curr_lane == target_lane:
            if dist_to_target_lane > 1:
                # 计算点到目标线上的投影 并加30m
                pro_s = target_lst.project(pos) + 30
                # 插值得到这个点的坐标 Point类型
                ip = target_lst.interpolate(pro_s)
                x1 = pos.x
                y1 = pos.y
                x2 = ip.x
                y2 = ip.y
            else:
                # 计算坐标点到中心线上的投影
                pro_s = curr_lst.project(pos)
                # 插值得到这个点的坐标 Point类型
                ip = curr_lst.interpolate(pro_s)
                segments = list(map(LineString, zip(curr_lst.coords[:-1], curr_lst.coords[1:])))
                # 使得插值得到的点p对每个线段计算距离，并排序
                segments.sort(key=lambda x: ip.distance(x))
                # line是距离ip最近的segment
                line = segments[0]
                x1, x2 = line.xy[0]
                y1, y2 = line.xy[1]
        else:
            # 计算点到目标线上的投影 并加30m
            pro_s = target_lst.project(pos)+30
            # 插值得到这个点的坐标 Point类型
            ip = target_lst.interpolate(pro_s)
            x1 = pos.x
            y1 = pos.y
            x2 = ip.x
            y2 = ip.y
        target_heading = math.atan2(y2 - y1, x2 - x1)
        # print(dist_to_target_lane, target_heading)
        return dist_to_target_lane, target_heading

    # 给一个坐标点pos,和可能所在的lane列表,找到pos所在的lane是什么
    def find_lane(self, pos, lanes):
        min_dist = 100000
        curr_lane = lanes[0]
        for lane in lanes:
            line = self.get_lane_central_curve(lane)
            dis = pos.distance(line)
            if (dis < min_dist):
                min_dist = dis
                curr_lane = lane
        return curr_lane

    def isin_lanes(self, pos, lanes, left_lane = None):
        if (left_lane != None): lanes = lanes.append(left_lane)
        min_dist = 100000
        curr_lane = lanes[0]
        for lane in lanes:
            line = self.get_lane_central_curve(lane)
            dis = pos.distance(line)
            if (dis < min_dist):
                min_dist = dis
                curr_lane = lane
        return curr_lane, min_dist

    def get_junctions(self) -> List[str]:
        """
        Get a list of all junction IDs on the HD Map

        :returns: list of junction IDs
        :rtype: List[str]
        """
        return list(self.__junctions.keys())

    def get_junction_by_id(self, j_id: str) -> Junction:
        """
        Get a specific junction object based on ID

        :param str j_id: ID of the junction interested in

        :returns: junction object
        :rtype: Junction
        """
        return self.__junctions[j_id]

    def get_lanes(self) -> List[str]:
        """
        Get a list of all lane IDs on the HD Map

        :returns: list of lane IDs
        :rtype: List[str]
        """
        return list(self.__lanes.keys())

    def get_lane_by_id(self, l_id: str) -> Lane:
        """
        Get a specific junction object based on ID

        :param str l_id: ID of the lane interested in

        :returns: lane object
        :rtype: Lane
        """
        return self.__lanes[l_id]

    def get_crosswalks(self) -> List[str]:
        """
        Get a list of all crosswalk IDs on the HD Map

        :returns: list of crosswalk IDs
        :rtype: List[str]
        """
        return list(self.__crosswalk.keys())

    def get_crosswalk_by_id(self, cw_id: str) -> Crosswalk:
        """
        Get a specific crosswalk object based on ID

        :param str cw_id: ID of the crosswalk interested in

        :returns: crosswalk object
        :rtype: Crosswalk
        """
        return self.__crosswalk[cw_id]

    def get_signals(self) -> List[str]:
        """
        Get a list of all signal IDs on the HD Map

        :returns: list of signal IDs
        :rtype: List[str]
        """
        return list(self.__signals.keys())

    def get_signal_by_id(self, s_id: str) -> Signal:
        """
        Get a specific signal object based on ID

        :param str s_id: ID of the signal interested in

        :returns: signal object
        :rtype: Signal
        """
        return self.__signals[s_id]

    def get_stop_signs(self) -> List[str]:
        """
        Get a list of all stop sign IDs on the HD Map

        :returns: list of stop sign IDs
        :rtype: List[str]
        """
        return list(self.__stop_signs.keys())

    def get_stop_sign_by_id(self, ss_id: str) -> StopSign:
        """
        Get a specific stop sign object based on ID

        :param str ss_id: ID of the stop sign interested in

        :returns: stop sign object
        :rtype: StopSign
        """
        return self.__stop_signs[ss_id]

    def get_lanes_not_in_junction(self) -> Set[str]:
        """
        Get the set of all lanes that are not in the junction.

        :returns: ID of lanes who is not in a junction
        :rtype: Set[str]
        """
        lanes = set(self.get_lanes())
        for junc in self.__lanes_at_junction:
            jlanes = set(self.__lanes_at_junction[junc])
            lanes = lanes - jlanes
        return lanes

    def get_path_from(self, lane_id: str) -> dict:
        """
        Get possible paths starting from a specified lane

        :param str lane_id: ID of the starting lane

        :returns: list of possible paths, each consisten multiple lanes
        :rtype: List[List[str]]
        """
        '''
        enum LaneTurn {
            NO_TURN = 1;
            LEFT_TURN = 2;
            RIGHT_TURN = 3;
            U_TURN = 4;
        };
        '''
        target_lanes = self.get_lanes_not_in_junction()
        reachable = self.__get_reachable_from(lane_id)
        allowed_path =  [p for p in reachable if p[-1] in target_lanes]
        straight_path = []
        right_path = []
        left_path = []

        for p in allowed_path:
            # 得到p的所有lane的朝向
            p_turn = []
            for l in p:
                lane = self.get_lane_by_id(l)
                p_turn.append(lane.turn)
            # left
            # 如果path中有左转的
            if 2 in p_turn:
                index = p_turn.index(2)
                # 将左转的lane名字加入left_path
                l_path = [p[index]]
                # 将前后lane也加入
                if index-1 >= 0: l_path.insert(0,p[index-1])
                if index+1 <= len(p_turn): l_path.append(p[index+1])
                left_path.append(l_path)
            if 3 in p_turn:
                index = p_turn.index(3)
                r_path = [p[index]]
                if index - 1 >= 0: r_path.insert(0, p[index - 1])
                if index + 1 < len(p_turn): r_path.append(p[index + 1])
                right_path.append(r_path)
            if 2 not in p_turn and 3 not in p_turn:
                straight_path.append(p)
        return {'straight':straight_path, 'left':left_path, 'right':right_path, 'all': allowed_path}

    def __get_reachable_from(self, lane_id: str, depth=5) -> List[List[str]]:
        """
        Recursive method to compute paths no more than ``depth`` lanes from
        the starting lane.

        :param str lane_id: ID of the starting lane
        :param int depth: maximum number of lanes traveled

        :returns: list of possible paths
        :rtype: List[List[str]]
        """

        if depth == 1:
            return [[lane_id]]
        result = list()
        for u, v in self.__lane_nx.edges(lane_id):
            result.append([u, v])
            for rp in self.__get_reachable_from(v, depth-1):
                result.append([u] + rp)
        return result


    def get_lane_turn(self, direction:str):
        if(direction == 'straight'):
            return self.noturn_lanes
        elif(direction == 'left'):
            return self.leftturn_lanes
        elif(direction == 'right'):
            return self.rightturn_lanes


    def get_lane_turn_by_id(self, lane_id:str):
        """
        get the LaneTurn from map_lane.proto
        """
        lane = self.__lanes[lane_id]
        '''
        enum LaneTurn {
            NO_TURN = 1;
            LEFT_TURN = 2;
            RIGHT_TURN = 3;
            U_TURN = 4;
        };
        '''
        return lane.turn


    def get_lane_direction_by_id(self, lane_id:str):
        """
        get the LaneDirection from map_lane.proto
        """
        lane = self.__lanes[lane_id]
        '''
        enum LaneDirection
        {
            FORWARD = 1;
            BACKWARD = 2;
            BIDIRECTION = 3;
        }
        '''
        return lane.direction

    def get_lane_boundary_by_id(self, lane_id:str):
        """
        get the LaneBoundary from map_lane.proto
        """
        lane = self.__lanes[lane_id]
        '''
        enum Type {
            UNKNOWN = 0;
            DOTTED_YELLOW = 1;
            DOTTED_WHITE = 2;
            SOLID_YELLOW = 3;
            SOLID_WHITE = 4;
            DOUBLE_YELLOW = 5;
            CURB = 6;
        };
        '''
        left_boundary_list = lane.left_boundary.boundary_type
        right_boundary_list = lane.right_boundary.boundary_type
        return {'left':left_boundary_list[0].types[0], 'right':right_boundary_list[0].types[0]}


    def get_signal_control_lane(self, lane_id:str) -> str:
        '''
          enum Color {
        UNKNOWN = 0;
        RED = 1;
        YELLOW = 2;
        GREEN = 3;
        BLACK = 4;
        FLASH_GREEN = 5;
        '''
        sig_lane = self.__lanes_controlled_by_signal
        for sig in self.__signals.keys():
            lane_list = sig_lane[sig]
            if lane_id in lane_list:
                return sig
        return ""

    def get_sign_control_lane(self, lane_id:str) -> str:
        # sign0--lane23
        # sign1--lane25
        if (lane_id == "lane_23"): return "stopsign_0"
        elif (lane_id == "lane_25"): return "stopsign_1"
        else: return ""


    def get_stopline_of_signal(self, sig_id: str) -> LineString:
        signal = self.__signals[sig_id]
        points = signal.stop_line[0].segment[0].line_segment
        line = LineString([[x.x, x.y] for x in points.point])
        return line


    def get_stopline_of_stop(self, stop_id: str) -> LineString:
        sign = self.__stop_signs[stop_id]
        points = sign.stop_line[0].segment[0].line_segment
        line = LineString([[x.x, x.y] for x in points.point])
        return line

    def get_left_neighbor_forward_lane(self, lane_id:str):
        lane = self.get_lane_by_id(lane_id)
        if(len(lane.left_neighbor_forward_lane_id) == 0): return None
        else: return lane.left_neighbor_forward_lane_id[0].id

    def get_right_neighbor_forward_lane(self, lane_id:str):
        lane = self.get_lane_by_id(lane_id)
        if(len(lane.right_neighbor_forward_lane_id) == 0): return None
        else: return lane.right_neighbor_forward_lane_id[0].id

    def get_postion_relation(self, laneheading:float, pos1:Point, pos2:Point):
        # 获取当前的1指向2的弧度值
        direction = math.atan2(pos2.y - pos1.y, pos2.x - pos1.x)
        # 求laneheading和1->2的角度之差
        # 如果是锐角说明在2在1前面，如果是钝角说明1在2前面
        rela_direc_cos = math.cos(laneheading - direction)

        # 1在2前面返回True
        if rela_direc_cos >= 0: return False
        else: return True


    def get_junction_polygon(self, junc_id) -> Polygon:
        junc = self.__junctions[junc_id]
        points = junc.polygon
        polygon = Polygon([[x.x, x.y] for x in points.point])
        return polygon