# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modules/dreamview/backend/teleop/proto/daemon_rpt.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

from  modules.common.proto import header_pb2 as modules_dot_common_dot_proto_dot_header__pb2

DESCRIPTOR = _descriptor.FileDescriptor(
  name='modules/dreamview/backend/teleop/proto/daemon_rpt.proto',
  package='modules.teleop.daemon',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n7modules/dreamview/backend/teleop/proto/daemon_rpt.proto\x12\x15modules.teleop.daemon\x1a!modules/common/proto/header.proto\"D\n\tDaemonRpt\x12%\n\x06header\x18\x01 \x01(\x0b\x32\x15.apollo.common.Header\x12\x10\n\x08services\x18\x02 \x03(\t')
  ,
  dependencies=[modules_dot_common_dot_proto_dot_header__pb2.DESCRIPTOR,])




_DAEMONRPT = _descriptor.Descriptor(
  name='DaemonRpt',
  full_name='modules.teleop.daemon.DaemonRpt',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='header', full_name='modules.teleop.daemon.DaemonRpt.header', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='services', full_name='modules.teleop.daemon.DaemonRpt.services', index=1,
      number=2, type=9, cpp_type=9, label=3,
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
  serialized_start=117,
  serialized_end=185,
)

_DAEMONRPT.fields_by_name['header'].message_type = modules_dot_common_dot_proto_dot_header__pb2._HEADER
DESCRIPTOR.message_types_by_name['DaemonRpt'] = _DAEMONRPT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

DaemonRpt = _reflection.GeneratedProtocolMessageType('DaemonRpt', (_message.Message,), dict(
  DESCRIPTOR = _DAEMONRPT,
  __module__ = 'modules.dreamview.backend.teleop.proto.daemon_rpt_pb2'
  # @@protoc_insertion_point(class_scope:modules.teleop.daemon.DaemonRpt)
  ))
_sym_db.RegisterMessage(DaemonRpt)


# @@protoc_insertion_point(module_scope)
