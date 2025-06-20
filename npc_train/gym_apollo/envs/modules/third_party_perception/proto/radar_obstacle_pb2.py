# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modules/third_party_perception/proto/radar_obstacle.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

from gym_apollo.envs.modules.common.proto import geometry_pb2 as modules_dot_common_dot_proto_dot_geometry__pb2, \
    error_code_pb2 as modules_dot_common_dot_proto_dot_error__code__pb2, \
    header_pb2 as modules_dot_common_dot_proto_dot_header__pb2

DESCRIPTOR = _descriptor.FileDescriptor(
  name='modules/third_party_perception/proto/radar_obstacle.proto',
  package='apollo.third_party_perception',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n9modules/third_party_perception/proto/radar_obstacle.proto\x12\x1d\x61pollo.third_party_perception\x1a%modules/common/proto/error_code.proto\x1a!modules/common/proto/header.proto\x1a#modules/common/proto/geometry.proto\"\xef\x02\n\rRadarObstacle\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x31\n\x11relative_position\x18\x02 \x01(\x0b\x32\x16.apollo.common.Point3D\x12\x31\n\x11relative_velocity\x18\x03 \x01(\x0b\x32\x16.apollo.common.Point3D\x12\x0b\n\x03rcs\x18\x04 \x01(\x01\x12\x0f\n\x07movable\x18\x05 \x01(\x08\x12\r\n\x05width\x18\x06 \x01(\x01\x12\x0e\n\x06length\x18\x07 \x01(\x01\x12\x0e\n\x06height\x18\x08 \x01(\x01\x12\r\n\x05theta\x18\t \x01(\x01\x12\x31\n\x11\x61\x62solute_position\x18\n \x01(\x0b\x32\x16.apollo.common.Point3D\x12\x31\n\x11\x61\x62solute_velocity\x18\x0b \x01(\x0b\x32\x16.apollo.common.Point3D\x12\r\n\x05\x63ount\x18\x0c \x01(\x05\x12\x1b\n\x13moving_frames_count\x18\r \x01(\x05\"\xa7\x02\n\x0eRadarObstacles\x12X\n\x0eradar_obstacle\x18\x01 \x03(\x0b\x32@.apollo.third_party_perception.RadarObstacles.RadarObstacleEntry\x12%\n\x06header\x18\x02 \x01(\x0b\x32\x15.apollo.common.Header\x12\x30\n\nerror_code\x18\x03 \x01(\x0e\x32\x18.apollo.common.ErrorCode:\x02OK\x1a\x62\n\x12RadarObstacleEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12;\n\x05value\x18\x02 \x01(\x0b\x32,.apollo.third_party_perception.RadarObstacle:\x02\x38\x01')
  ,
  dependencies=[modules_dot_common_dot_proto_dot_error__code__pb2.DESCRIPTOR,modules_dot_common_dot_proto_dot_header__pb2.DESCRIPTOR,modules_dot_common_dot_proto_dot_geometry__pb2.DESCRIPTOR,])




_RADAROBSTACLE = _descriptor.Descriptor(
  name='RadarObstacle',
  full_name='apollo.third_party_perception.RadarObstacle',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='apollo.third_party_perception.RadarObstacle.id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='relative_position', full_name='apollo.third_party_perception.RadarObstacle.relative_position', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='relative_velocity', full_name='apollo.third_party_perception.RadarObstacle.relative_velocity', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rcs', full_name='apollo.third_party_perception.RadarObstacle.rcs', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='movable', full_name='apollo.third_party_perception.RadarObstacle.movable', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='width', full_name='apollo.third_party_perception.RadarObstacle.width', index=5,
      number=6, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='length', full_name='apollo.third_party_perception.RadarObstacle.length', index=6,
      number=7, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='apollo.third_party_perception.RadarObstacle.height', index=7,
      number=8, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='theta', full_name='apollo.third_party_perception.RadarObstacle.theta', index=8,
      number=9, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='absolute_position', full_name='apollo.third_party_perception.RadarObstacle.absolute_position', index=9,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='absolute_velocity', full_name='apollo.third_party_perception.RadarObstacle.absolute_velocity', index=10,
      number=11, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='count', full_name='apollo.third_party_perception.RadarObstacle.count', index=11,
      number=12, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='moving_frames_count', full_name='apollo.third_party_perception.RadarObstacle.moving_frames_count', index=12,
      number=13, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=204,
  serialized_end=571,
)


_RADAROBSTACLES_RADAROBSTACLEENTRY = _descriptor.Descriptor(
  name='RadarObstacleEntry',
  full_name='apollo.third_party_perception.RadarObstacles.RadarObstacleEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='apollo.third_party_perception.RadarObstacles.RadarObstacleEntry.key', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='apollo.third_party_perception.RadarObstacles.RadarObstacleEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
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
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=771,
  serialized_end=869,
)

_RADAROBSTACLES = _descriptor.Descriptor(
  name='RadarObstacles',
  full_name='apollo.third_party_perception.RadarObstacles',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='radar_obstacle', full_name='apollo.third_party_perception.RadarObstacles.radar_obstacle', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='header', full_name='apollo.third_party_perception.RadarObstacles.header', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='error_code', full_name='apollo.third_party_perception.RadarObstacles.error_code', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_RADAROBSTACLES_RADAROBSTACLEENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=574,
  serialized_end=869,
)

_RADAROBSTACLE.fields_by_name['relative_position'].message_type = modules_dot_common_dot_proto_dot_geometry__pb2._POINT3D
_RADAROBSTACLE.fields_by_name['relative_velocity'].message_type = modules_dot_common_dot_proto_dot_geometry__pb2._POINT3D
_RADAROBSTACLE.fields_by_name['absolute_position'].message_type = modules_dot_common_dot_proto_dot_geometry__pb2._POINT3D
_RADAROBSTACLE.fields_by_name['absolute_velocity'].message_type = modules_dot_common_dot_proto_dot_geometry__pb2._POINT3D
_RADAROBSTACLES_RADAROBSTACLEENTRY.fields_by_name['value'].message_type = _RADAROBSTACLE
_RADAROBSTACLES_RADAROBSTACLEENTRY.containing_type = _RADAROBSTACLES
_RADAROBSTACLES.fields_by_name['radar_obstacle'].message_type = _RADAROBSTACLES_RADAROBSTACLEENTRY
_RADAROBSTACLES.fields_by_name['header'].message_type = modules_dot_common_dot_proto_dot_header__pb2._HEADER
_RADAROBSTACLES.fields_by_name['error_code'].enum_type = modules_dot_common_dot_proto_dot_error__code__pb2._ERRORCODE
DESCRIPTOR.message_types_by_name['RadarObstacle'] = _RADAROBSTACLE
DESCRIPTOR.message_types_by_name['RadarObstacles'] = _RADAROBSTACLES
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RadarObstacle = _reflection.GeneratedProtocolMessageType('RadarObstacle', (_message.Message,), dict(
  DESCRIPTOR = _RADAROBSTACLE,
  __module__ = 'modules.third_party_perception.proto.radar_obstacle_pb2'
  # @@protoc_insertion_point(class_scope:apollo.third_party_perception.RadarObstacle)
  ))
_sym_db.RegisterMessage(RadarObstacle)

RadarObstacles = _reflection.GeneratedProtocolMessageType('RadarObstacles', (_message.Message,), dict(

  RadarObstacleEntry = _reflection.GeneratedProtocolMessageType('RadarObstacleEntry', (_message.Message,), dict(
    DESCRIPTOR = _RADAROBSTACLES_RADAROBSTACLEENTRY,
    __module__ = 'modules.third_party_perception.proto.radar_obstacle_pb2'
    # @@protoc_insertion_point(class_scope:apollo.third_party_perception.RadarObstacles.RadarObstacleEntry)
    ))
  ,
  DESCRIPTOR = _RADAROBSTACLES,
  __module__ = 'modules.third_party_perception.proto.radar_obstacle_pb2'
  # @@protoc_insertion_point(class_scope:apollo.third_party_perception.RadarObstacles)
  ))
_sym_db.RegisterMessage(RadarObstacles)
_sym_db.RegisterMessage(RadarObstacles.RadarObstacleEntry)


_RADAROBSTACLES_RADAROBSTACLEENTRY._options = None
# @@protoc_insertion_point(module_scope)
