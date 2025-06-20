# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modules/map/proto/map_overlap.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

from gym_apollo.envs.modules.map.proto import map_id_pb2 as modules_dot_map_dot_proto_dot_map__id__pb2, \
    map_geometry_pb2 as modules_dot_map_dot_proto_dot_map__geometry__pb2

DESCRIPTOR = _descriptor.FileDescriptor(
  name='modules/map/proto/map_overlap.proto',
  package='apollo.hdmap',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n#modules/map/proto/map_overlap.proto\x12\x0c\x61pollo.hdmap\x1a\x1emodules/map/proto/map_id.proto\x1a$modules/map/proto/map_geometry.proto\"p\n\x0fLaneOverlapInfo\x12\x0f\n\x07start_s\x18\x01 \x01(\x01\x12\r\n\x05\x65nd_s\x18\x02 \x01(\x01\x12\x10\n\x08is_merge\x18\x03 \x01(\x08\x12+\n\x11region_overlap_id\x18\x04 \x01(\x0b\x32\x10.apollo.hdmap.Id\"\x13\n\x11SignalOverlapInfo\"\x15\n\x13StopSignOverlapInfo\"C\n\x14\x43rosswalkOverlapInfo\x12+\n\x11region_overlap_id\x18\x01 \x01(\x0b\x32\x10.apollo.hdmap.Id\"\x15\n\x13JunctionOverlapInfo\"\x12\n\x10YieldOverlapInfo\"\x16\n\x14\x43learAreaOverlapInfo\"\x16\n\x14SpeedBumpOverlapInfo\"\x19\n\x17ParkingSpaceOverlapInfo\"\x18\n\x16PNCJunctionOverlapInfo\"\x10\n\x0eRSUOverlapInfo\"Y\n\x11RegionOverlapInfo\x12\x1c\n\x02id\x18\x01 \x01(\x0b\x32\x10.apollo.hdmap.Id\x12&\n\x07polygon\x18\x02 \x03(\x0b\x32\x15.apollo.hdmap.Polygon\"\xaf\x06\n\x11ObjectOverlapInfo\x12\x1c\n\x02id\x18\x01 \x01(\x0b\x32\x10.apollo.hdmap.Id\x12:\n\x11lane_overlap_info\x18\x03 \x01(\x0b\x32\x1d.apollo.hdmap.LaneOverlapInfoH\x00\x12>\n\x13signal_overlap_info\x18\x04 \x01(\x0b\x32\x1f.apollo.hdmap.SignalOverlapInfoH\x00\x12\x43\n\x16stop_sign_overlap_info\x18\x05 \x01(\x0b\x32!.apollo.hdmap.StopSignOverlapInfoH\x00\x12\x44\n\x16\x63rosswalk_overlap_info\x18\x06 \x01(\x0b\x32\".apollo.hdmap.CrosswalkOverlapInfoH\x00\x12\x42\n\x15junction_overlap_info\x18\x07 \x01(\x0b\x32!.apollo.hdmap.JunctionOverlapInfoH\x00\x12\x41\n\x17yield_sign_overlap_info\x18\x08 \x01(\x0b\x32\x1e.apollo.hdmap.YieldOverlapInfoH\x00\x12\x45\n\x17\x63lear_area_overlap_info\x18\t \x01(\x0b\x32\".apollo.hdmap.ClearAreaOverlapInfoH\x00\x12\x45\n\x17speed_bump_overlap_info\x18\n \x01(\x0b\x32\".apollo.hdmap.SpeedBumpOverlapInfoH\x00\x12K\n\x1aparking_space_overlap_info\x18\x0b \x01(\x0b\x32%.apollo.hdmap.ParkingSpaceOverlapInfoH\x00\x12I\n\x19pnc_junction_overlap_info\x18\x0c \x01(\x0b\x32$.apollo.hdmap.PNCJunctionOverlapInfoH\x00\x12\x38\n\x10rsu_overlap_info\x18\r \x01(\x0b\x32\x1c.apollo.hdmap.RSUOverlapInfoH\x00\x42\x0e\n\x0coverlap_info\"\x91\x01\n\x07Overlap\x12\x1c\n\x02id\x18\x01 \x01(\x0b\x32\x10.apollo.hdmap.Id\x12/\n\x06object\x18\x02 \x03(\x0b\x32\x1f.apollo.hdmap.ObjectOverlapInfo\x12\x37\n\x0eregion_overlap\x18\x03 \x03(\x0b\x32\x1f.apollo.hdmap.RegionOverlapInfo')
  ,
  dependencies=[modules_dot_map_dot_proto_dot_map__id__pb2.DESCRIPTOR,modules_dot_map_dot_proto_dot_map__geometry__pb2.DESCRIPTOR,])




_LANEOVERLAPINFO = _descriptor.Descriptor(
  name='LaneOverlapInfo',
  full_name='apollo.hdmap.LaneOverlapInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='start_s', full_name='apollo.hdmap.LaneOverlapInfo.start_s', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='end_s', full_name='apollo.hdmap.LaneOverlapInfo.end_s', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='is_merge', full_name='apollo.hdmap.LaneOverlapInfo.is_merge', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='region_overlap_id', full_name='apollo.hdmap.LaneOverlapInfo.region_overlap_id', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=123,
  serialized_end=235,
)


_SIGNALOVERLAPINFO = _descriptor.Descriptor(
  name='SignalOverlapInfo',
  full_name='apollo.hdmap.SignalOverlapInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=237,
  serialized_end=256,
)


_STOPSIGNOVERLAPINFO = _descriptor.Descriptor(
  name='StopSignOverlapInfo',
  full_name='apollo.hdmap.StopSignOverlapInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=258,
  serialized_end=279,
)


_CROSSWALKOVERLAPINFO = _descriptor.Descriptor(
  name='CrosswalkOverlapInfo',
  full_name='apollo.hdmap.CrosswalkOverlapInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='region_overlap_id', full_name='apollo.hdmap.CrosswalkOverlapInfo.region_overlap_id', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=281,
  serialized_end=348,
)


_JUNCTIONOVERLAPINFO = _descriptor.Descriptor(
  name='JunctionOverlapInfo',
  full_name='apollo.hdmap.JunctionOverlapInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=350,
  serialized_end=371,
)


_YIELDOVERLAPINFO = _descriptor.Descriptor(
  name='YieldOverlapInfo',
  full_name='apollo.hdmap.YieldOverlapInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=373,
  serialized_end=391,
)


_CLEARAREAOVERLAPINFO = _descriptor.Descriptor(
  name='ClearAreaOverlapInfo',
  full_name='apollo.hdmap.ClearAreaOverlapInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=393,
  serialized_end=415,
)


_SPEEDBUMPOVERLAPINFO = _descriptor.Descriptor(
  name='SpeedBumpOverlapInfo',
  full_name='apollo.hdmap.SpeedBumpOverlapInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=417,
  serialized_end=439,
)


_PARKINGSPACEOVERLAPINFO = _descriptor.Descriptor(
  name='ParkingSpaceOverlapInfo',
  full_name='apollo.hdmap.ParkingSpaceOverlapInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=441,
  serialized_end=466,
)


_PNCJUNCTIONOVERLAPINFO = _descriptor.Descriptor(
  name='PNCJunctionOverlapInfo',
  full_name='apollo.hdmap.PNCJunctionOverlapInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=468,
  serialized_end=492,
)


_RSUOVERLAPINFO = _descriptor.Descriptor(
  name='RSUOverlapInfo',
  full_name='apollo.hdmap.RSUOverlapInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=494,
  serialized_end=510,
)


_REGIONOVERLAPINFO = _descriptor.Descriptor(
  name='RegionOverlapInfo',
  full_name='apollo.hdmap.RegionOverlapInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='apollo.hdmap.RegionOverlapInfo.id', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='polygon', full_name='apollo.hdmap.RegionOverlapInfo.polygon', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=512,
  serialized_end=601,
)


_OBJECTOVERLAPINFO = _descriptor.Descriptor(
  name='ObjectOverlapInfo',
  full_name='apollo.hdmap.ObjectOverlapInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='apollo.hdmap.ObjectOverlapInfo.id', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='lane_overlap_info', full_name='apollo.hdmap.ObjectOverlapInfo.lane_overlap_info', index=1,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='signal_overlap_info', full_name='apollo.hdmap.ObjectOverlapInfo.signal_overlap_info', index=2,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stop_sign_overlap_info', full_name='apollo.hdmap.ObjectOverlapInfo.stop_sign_overlap_info', index=3,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='crosswalk_overlap_info', full_name='apollo.hdmap.ObjectOverlapInfo.crosswalk_overlap_info', index=4,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='junction_overlap_info', full_name='apollo.hdmap.ObjectOverlapInfo.junction_overlap_info', index=5,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='yield_sign_overlap_info', full_name='apollo.hdmap.ObjectOverlapInfo.yield_sign_overlap_info', index=6,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='clear_area_overlap_info', full_name='apollo.hdmap.ObjectOverlapInfo.clear_area_overlap_info', index=7,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='speed_bump_overlap_info', full_name='apollo.hdmap.ObjectOverlapInfo.speed_bump_overlap_info', index=8,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='parking_space_overlap_info', full_name='apollo.hdmap.ObjectOverlapInfo.parking_space_overlap_info', index=9,
      number=11, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pnc_junction_overlap_info', full_name='apollo.hdmap.ObjectOverlapInfo.pnc_junction_overlap_info', index=10,
      number=12, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rsu_overlap_info', full_name='apollo.hdmap.ObjectOverlapInfo.rsu_overlap_info', index=11,
      number=13, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='overlap_info', full_name='apollo.hdmap.ObjectOverlapInfo.overlap_info',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=604,
  serialized_end=1419,
)


_OVERLAP = _descriptor.Descriptor(
  name='Overlap',
  full_name='apollo.hdmap.Overlap',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='apollo.hdmap.Overlap.id', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='object', full_name='apollo.hdmap.Overlap.object', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='region_overlap', full_name='apollo.hdmap.Overlap.region_overlap', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1422,
  serialized_end=1567,
)

_LANEOVERLAPINFO.fields_by_name['region_overlap_id'].message_type = modules_dot_map_dot_proto_dot_map__id__pb2._ID
_CROSSWALKOVERLAPINFO.fields_by_name['region_overlap_id'].message_type = modules_dot_map_dot_proto_dot_map__id__pb2._ID
_REGIONOVERLAPINFO.fields_by_name['id'].message_type = modules_dot_map_dot_proto_dot_map__id__pb2._ID
_REGIONOVERLAPINFO.fields_by_name['polygon'].message_type = modules_dot_map_dot_proto_dot_map__geometry__pb2._POLYGON
_OBJECTOVERLAPINFO.fields_by_name['id'].message_type = modules_dot_map_dot_proto_dot_map__id__pb2._ID
_OBJECTOVERLAPINFO.fields_by_name['lane_overlap_info'].message_type = _LANEOVERLAPINFO
_OBJECTOVERLAPINFO.fields_by_name['signal_overlap_info'].message_type = _SIGNALOVERLAPINFO
_OBJECTOVERLAPINFO.fields_by_name['stop_sign_overlap_info'].message_type = _STOPSIGNOVERLAPINFO
_OBJECTOVERLAPINFO.fields_by_name['crosswalk_overlap_info'].message_type = _CROSSWALKOVERLAPINFO
_OBJECTOVERLAPINFO.fields_by_name['junction_overlap_info'].message_type = _JUNCTIONOVERLAPINFO
_OBJECTOVERLAPINFO.fields_by_name['yield_sign_overlap_info'].message_type = _YIELDOVERLAPINFO
_OBJECTOVERLAPINFO.fields_by_name['clear_area_overlap_info'].message_type = _CLEARAREAOVERLAPINFO
_OBJECTOVERLAPINFO.fields_by_name['speed_bump_overlap_info'].message_type = _SPEEDBUMPOVERLAPINFO
_OBJECTOVERLAPINFO.fields_by_name['parking_space_overlap_info'].message_type = _PARKINGSPACEOVERLAPINFO
_OBJECTOVERLAPINFO.fields_by_name['pnc_junction_overlap_info'].message_type = _PNCJUNCTIONOVERLAPINFO
_OBJECTOVERLAPINFO.fields_by_name['rsu_overlap_info'].message_type = _RSUOVERLAPINFO
_OBJECTOVERLAPINFO.oneofs_by_name['overlap_info'].fields.append(
  _OBJECTOVERLAPINFO.fields_by_name['lane_overlap_info'])
_OBJECTOVERLAPINFO.fields_by_name['lane_overlap_info'].containing_oneof = _OBJECTOVERLAPINFO.oneofs_by_name['overlap_info']
_OBJECTOVERLAPINFO.oneofs_by_name['overlap_info'].fields.append(
  _OBJECTOVERLAPINFO.fields_by_name['signal_overlap_info'])
_OBJECTOVERLAPINFO.fields_by_name['signal_overlap_info'].containing_oneof = _OBJECTOVERLAPINFO.oneofs_by_name['overlap_info']
_OBJECTOVERLAPINFO.oneofs_by_name['overlap_info'].fields.append(
  _OBJECTOVERLAPINFO.fields_by_name['stop_sign_overlap_info'])
_OBJECTOVERLAPINFO.fields_by_name['stop_sign_overlap_info'].containing_oneof = _OBJECTOVERLAPINFO.oneofs_by_name['overlap_info']
_OBJECTOVERLAPINFO.oneofs_by_name['overlap_info'].fields.append(
  _OBJECTOVERLAPINFO.fields_by_name['crosswalk_overlap_info'])
_OBJECTOVERLAPINFO.fields_by_name['crosswalk_overlap_info'].containing_oneof = _OBJECTOVERLAPINFO.oneofs_by_name['overlap_info']
_OBJECTOVERLAPINFO.oneofs_by_name['overlap_info'].fields.append(
  _OBJECTOVERLAPINFO.fields_by_name['junction_overlap_info'])
_OBJECTOVERLAPINFO.fields_by_name['junction_overlap_info'].containing_oneof = _OBJECTOVERLAPINFO.oneofs_by_name['overlap_info']
_OBJECTOVERLAPINFO.oneofs_by_name['overlap_info'].fields.append(
  _OBJECTOVERLAPINFO.fields_by_name['yield_sign_overlap_info'])
_OBJECTOVERLAPINFO.fields_by_name['yield_sign_overlap_info'].containing_oneof = _OBJECTOVERLAPINFO.oneofs_by_name['overlap_info']
_OBJECTOVERLAPINFO.oneofs_by_name['overlap_info'].fields.append(
  _OBJECTOVERLAPINFO.fields_by_name['clear_area_overlap_info'])
_OBJECTOVERLAPINFO.fields_by_name['clear_area_overlap_info'].containing_oneof = _OBJECTOVERLAPINFO.oneofs_by_name['overlap_info']
_OBJECTOVERLAPINFO.oneofs_by_name['overlap_info'].fields.append(
  _OBJECTOVERLAPINFO.fields_by_name['speed_bump_overlap_info'])
_OBJECTOVERLAPINFO.fields_by_name['speed_bump_overlap_info'].containing_oneof = _OBJECTOVERLAPINFO.oneofs_by_name['overlap_info']
_OBJECTOVERLAPINFO.oneofs_by_name['overlap_info'].fields.append(
  _OBJECTOVERLAPINFO.fields_by_name['parking_space_overlap_info'])
_OBJECTOVERLAPINFO.fields_by_name['parking_space_overlap_info'].containing_oneof = _OBJECTOVERLAPINFO.oneofs_by_name['overlap_info']
_OBJECTOVERLAPINFO.oneofs_by_name['overlap_info'].fields.append(
  _OBJECTOVERLAPINFO.fields_by_name['pnc_junction_overlap_info'])
_OBJECTOVERLAPINFO.fields_by_name['pnc_junction_overlap_info'].containing_oneof = _OBJECTOVERLAPINFO.oneofs_by_name['overlap_info']
_OBJECTOVERLAPINFO.oneofs_by_name['overlap_info'].fields.append(
  _OBJECTOVERLAPINFO.fields_by_name['rsu_overlap_info'])
_OBJECTOVERLAPINFO.fields_by_name['rsu_overlap_info'].containing_oneof = _OBJECTOVERLAPINFO.oneofs_by_name['overlap_info']
_OVERLAP.fields_by_name['id'].message_type = modules_dot_map_dot_proto_dot_map__id__pb2._ID
_OVERLAP.fields_by_name['object'].message_type = _OBJECTOVERLAPINFO
_OVERLAP.fields_by_name['region_overlap'].message_type = _REGIONOVERLAPINFO
DESCRIPTOR.message_types_by_name['LaneOverlapInfo'] = _LANEOVERLAPINFO
DESCRIPTOR.message_types_by_name['SignalOverlapInfo'] = _SIGNALOVERLAPINFO
DESCRIPTOR.message_types_by_name['StopSignOverlapInfo'] = _STOPSIGNOVERLAPINFO
DESCRIPTOR.message_types_by_name['CrosswalkOverlapInfo'] = _CROSSWALKOVERLAPINFO
DESCRIPTOR.message_types_by_name['JunctionOverlapInfo'] = _JUNCTIONOVERLAPINFO
DESCRIPTOR.message_types_by_name['YieldOverlapInfo'] = _YIELDOVERLAPINFO
DESCRIPTOR.message_types_by_name['ClearAreaOverlapInfo'] = _CLEARAREAOVERLAPINFO
DESCRIPTOR.message_types_by_name['SpeedBumpOverlapInfo'] = _SPEEDBUMPOVERLAPINFO
DESCRIPTOR.message_types_by_name['ParkingSpaceOverlapInfo'] = _PARKINGSPACEOVERLAPINFO
DESCRIPTOR.message_types_by_name['PNCJunctionOverlapInfo'] = _PNCJUNCTIONOVERLAPINFO
DESCRIPTOR.message_types_by_name['RSUOverlapInfo'] = _RSUOVERLAPINFO
DESCRIPTOR.message_types_by_name['RegionOverlapInfo'] = _REGIONOVERLAPINFO
DESCRIPTOR.message_types_by_name['ObjectOverlapInfo'] = _OBJECTOVERLAPINFO
DESCRIPTOR.message_types_by_name['Overlap'] = _OVERLAP
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

LaneOverlapInfo = _reflection.GeneratedProtocolMessageType('LaneOverlapInfo', (_message.Message,), dict(
  DESCRIPTOR = _LANEOVERLAPINFO,
  __module__ = 'modules.map.proto.map_overlap_pb2'
  # @@protoc_insertion_point(class_scope:apollo.hdmap.LaneOverlapInfo)
  ))
_sym_db.RegisterMessage(LaneOverlapInfo)

SignalOverlapInfo = _reflection.GeneratedProtocolMessageType('SignalOverlapInfo', (_message.Message,), dict(
  DESCRIPTOR = _SIGNALOVERLAPINFO,
  __module__ = 'modules.map.proto.map_overlap_pb2'
  # @@protoc_insertion_point(class_scope:apollo.hdmap.SignalOverlapInfo)
  ))
_sym_db.RegisterMessage(SignalOverlapInfo)

StopSignOverlapInfo = _reflection.GeneratedProtocolMessageType('StopSignOverlapInfo', (_message.Message,), dict(
  DESCRIPTOR = _STOPSIGNOVERLAPINFO,
  __module__ = 'modules.map.proto.map_overlap_pb2'
  # @@protoc_insertion_point(class_scope:apollo.hdmap.StopSignOverlapInfo)
  ))
_sym_db.RegisterMessage(StopSignOverlapInfo)

CrosswalkOverlapInfo = _reflection.GeneratedProtocolMessageType('CrosswalkOverlapInfo', (_message.Message,), dict(
  DESCRIPTOR = _CROSSWALKOVERLAPINFO,
  __module__ = 'modules.map.proto.map_overlap_pb2'
  # @@protoc_insertion_point(class_scope:apollo.hdmap.CrosswalkOverlapInfo)
  ))
_sym_db.RegisterMessage(CrosswalkOverlapInfo)

JunctionOverlapInfo = _reflection.GeneratedProtocolMessageType('JunctionOverlapInfo', (_message.Message,), dict(
  DESCRIPTOR = _JUNCTIONOVERLAPINFO,
  __module__ = 'modules.map.proto.map_overlap_pb2'
  # @@protoc_insertion_point(class_scope:apollo.hdmap.JunctionOverlapInfo)
  ))
_sym_db.RegisterMessage(JunctionOverlapInfo)

YieldOverlapInfo = _reflection.GeneratedProtocolMessageType('YieldOverlapInfo', (_message.Message,), dict(
  DESCRIPTOR = _YIELDOVERLAPINFO,
  __module__ = 'modules.map.proto.map_overlap_pb2'
  # @@protoc_insertion_point(class_scope:apollo.hdmap.YieldOverlapInfo)
  ))
_sym_db.RegisterMessage(YieldOverlapInfo)

ClearAreaOverlapInfo = _reflection.GeneratedProtocolMessageType('ClearAreaOverlapInfo', (_message.Message,), dict(
  DESCRIPTOR = _CLEARAREAOVERLAPINFO,
  __module__ = 'modules.map.proto.map_overlap_pb2'
  # @@protoc_insertion_point(class_scope:apollo.hdmap.ClearAreaOverlapInfo)
  ))
_sym_db.RegisterMessage(ClearAreaOverlapInfo)

SpeedBumpOverlapInfo = _reflection.GeneratedProtocolMessageType('SpeedBumpOverlapInfo', (_message.Message,), dict(
  DESCRIPTOR = _SPEEDBUMPOVERLAPINFO,
  __module__ = 'modules.map.proto.map_overlap_pb2'
  # @@protoc_insertion_point(class_scope:apollo.hdmap.SpeedBumpOverlapInfo)
  ))
_sym_db.RegisterMessage(SpeedBumpOverlapInfo)

ParkingSpaceOverlapInfo = _reflection.GeneratedProtocolMessageType('ParkingSpaceOverlapInfo', (_message.Message,), dict(
  DESCRIPTOR = _PARKINGSPACEOVERLAPINFO,
  __module__ = 'modules.map.proto.map_overlap_pb2'
  # @@protoc_insertion_point(class_scope:apollo.hdmap.ParkingSpaceOverlapInfo)
  ))
_sym_db.RegisterMessage(ParkingSpaceOverlapInfo)

PNCJunctionOverlapInfo = _reflection.GeneratedProtocolMessageType('PNCJunctionOverlapInfo', (_message.Message,), dict(
  DESCRIPTOR = _PNCJUNCTIONOVERLAPINFO,
  __module__ = 'modules.map.proto.map_overlap_pb2'
  # @@protoc_insertion_point(class_scope:apollo.hdmap.PNCJunctionOverlapInfo)
  ))
_sym_db.RegisterMessage(PNCJunctionOverlapInfo)

RSUOverlapInfo = _reflection.GeneratedProtocolMessageType('RSUOverlapInfo', (_message.Message,), dict(
  DESCRIPTOR = _RSUOVERLAPINFO,
  __module__ = 'modules.map.proto.map_overlap_pb2'
  # @@protoc_insertion_point(class_scope:apollo.hdmap.RSUOverlapInfo)
  ))
_sym_db.RegisterMessage(RSUOverlapInfo)

RegionOverlapInfo = _reflection.GeneratedProtocolMessageType('RegionOverlapInfo', (_message.Message,), dict(
  DESCRIPTOR = _REGIONOVERLAPINFO,
  __module__ = 'modules.map.proto.map_overlap_pb2'
  # @@protoc_insertion_point(class_scope:apollo.hdmap.RegionOverlapInfo)
  ))
_sym_db.RegisterMessage(RegionOverlapInfo)

ObjectOverlapInfo = _reflection.GeneratedProtocolMessageType('ObjectOverlapInfo', (_message.Message,), dict(
  DESCRIPTOR = _OBJECTOVERLAPINFO,
  __module__ = 'modules.map.proto.map_overlap_pb2'
  # @@protoc_insertion_point(class_scope:apollo.hdmap.ObjectOverlapInfo)
  ))
_sym_db.RegisterMessage(ObjectOverlapInfo)

Overlap = _reflection.GeneratedProtocolMessageType('Overlap', (_message.Message,), dict(
  DESCRIPTOR = _OVERLAP,
  __module__ = 'modules.map.proto.map_overlap_pb2'
  # @@protoc_insertion_point(class_scope:apollo.hdmap.Overlap)
  ))
_sym_db.RegisterMessage(Overlap)


# @@protoc_insertion_point(module_scope)
