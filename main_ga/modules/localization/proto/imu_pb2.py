# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modules/localization/proto/imu.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

from  modules.common.proto import header_pb2 as modules_dot_common_dot_proto_dot_header__pb2
from  modules.localization.proto import pose_pb2 as modules_dot_localization_dot_proto_dot_pose__pb2

DESCRIPTOR = _descriptor.FileDescriptor(
  name='modules/localization/proto/imu.proto',
  package='apollo.localization',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n$modules/localization/proto/imu.proto\x12\x13\x61pollo.localization\x1a!modules/common/proto/header.proto\x1a%modules/localization/proto/pose.proto\"]\n\x0c\x43orrectedImu\x12%\n\x06header\x18\x01 \x01(\x0b\x32\x15.apollo.common.Header\x12&\n\x03imu\x18\x03 \x01(\x0b\x32\x19.apollo.localization.Pose')
  ,
  dependencies=[modules_dot_common_dot_proto_dot_header__pb2.DESCRIPTOR,modules_dot_localization_dot_proto_dot_pose__pb2.DESCRIPTOR,])




_CORRECTEDIMU = _descriptor.Descriptor(
  name='CorrectedImu',
  full_name='apollo.localization.CorrectedImu',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='header', full_name='apollo.localization.CorrectedImu.header', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='imu', full_name='apollo.localization.CorrectedImu.imu', index=1,
      number=3, type=11, cpp_type=10, label=1,
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
  serialized_start=135,
  serialized_end=228,
)

_CORRECTEDIMU.fields_by_name['header'].message_type = modules_dot_common_dot_proto_dot_header__pb2._HEADER
_CORRECTEDIMU.fields_by_name['imu'].message_type = modules_dot_localization_dot_proto_dot_pose__pb2._POSE
DESCRIPTOR.message_types_by_name['CorrectedImu'] = _CORRECTEDIMU
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

CorrectedImu = _reflection.GeneratedProtocolMessageType('CorrectedImu', (_message.Message,), dict(
  DESCRIPTOR = _CORRECTEDIMU,
  __module__ = 'modules.localization.proto.imu_pb2'
  # @@protoc_insertion_point(class_scope:apollo.localization.CorrectedImu)
  ))
_sym_db.RegisterMessage(CorrectedImu)


# @@protoc_insertion_point(module_scope)
