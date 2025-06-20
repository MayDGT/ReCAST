# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modules/drivers/lidar/proto/hesai.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

from gym_apollo.envs.modules.common.proto import header_pb2 as modules_dot_common_dot_proto_dot_header__pb2

DESCRIPTOR = _descriptor.FileDescriptor(
  name='modules/drivers/lidar/proto/hesai.proto',
  package='apollo.drivers.hesai',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\'modules/drivers/lidar/proto/hesai.proto\x12\x14\x61pollo.drivers.hesai\x1a!modules/common/proto/header.proto\".\n\x0fHesaiScanPacket\x12\r\n\x05stamp\x18\x01 \x01(\x04\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\x0c\"\xaf\x01\n\tHesaiScan\x12%\n\x06header\x18\x01 \x01(\x0b\x32\x15.apollo.common.Header\x12*\n\x05model\x18\x02 \x01(\x0e\x32\x1b.apollo.drivers.hesai.Model\x12:\n\x0b\x66iring_pkts\x18\x03 \x03(\x0b\x32%.apollo.drivers.hesai.HesaiScanPacket\x12\x13\n\x08\x62\x61setime\x18\x04 \x01(\x04:\x01\x30*/\n\x05Model\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x0c\n\x08HESAI40P\x10\x01\x12\x0b\n\x07HESAI64\x10\x02')
  ,
  dependencies=[modules_dot_common_dot_proto_dot_header__pb2.DESCRIPTOR,])

_MODEL = _descriptor.EnumDescriptor(
  name='Model',
  full_name='apollo.drivers.hesai.Model',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='HESAI40P', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='HESAI64', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=326,
  serialized_end=373,
)
_sym_db.RegisterEnumDescriptor(_MODEL)

Model = enum_type_wrapper.EnumTypeWrapper(_MODEL)
UNKNOWN = 0
HESAI40P = 1
HESAI64 = 2



_HESAISCANPACKET = _descriptor.Descriptor(
  name='HesaiScanPacket',
  full_name='apollo.drivers.hesai.HesaiScanPacket',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='stamp', full_name='apollo.drivers.hesai.HesaiScanPacket.stamp', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data', full_name='apollo.drivers.hesai.HesaiScanPacket.data', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
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
  serialized_start=100,
  serialized_end=146,
)


_HESAISCAN = _descriptor.Descriptor(
  name='HesaiScan',
  full_name='apollo.drivers.hesai.HesaiScan',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='header', full_name='apollo.drivers.hesai.HesaiScan.header', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model', full_name='apollo.drivers.hesai.HesaiScan.model', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='firing_pkts', full_name='apollo.drivers.hesai.HesaiScan.firing_pkts', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='basetime', full_name='apollo.drivers.hesai.HesaiScan.basetime', index=3,
      number=4, type=4, cpp_type=4, label=1,
      has_default_value=True, default_value=0,
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
  serialized_start=149,
  serialized_end=324,
)

_HESAISCAN.fields_by_name['header'].message_type = modules_dot_common_dot_proto_dot_header__pb2._HEADER
_HESAISCAN.fields_by_name['model'].enum_type = _MODEL
_HESAISCAN.fields_by_name['firing_pkts'].message_type = _HESAISCANPACKET
DESCRIPTOR.message_types_by_name['HesaiScanPacket'] = _HESAISCANPACKET
DESCRIPTOR.message_types_by_name['HesaiScan'] = _HESAISCAN
DESCRIPTOR.enum_types_by_name['Model'] = _MODEL
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

HesaiScanPacket = _reflection.GeneratedProtocolMessageType('HesaiScanPacket', (_message.Message,), dict(
  DESCRIPTOR = _HESAISCANPACKET,
  __module__ = 'modules.drivers.lidar.proto.hesai_pb2'
  # @@protoc_insertion_point(class_scope:apollo.drivers.hesai.HesaiScanPacket)
  ))
_sym_db.RegisterMessage(HesaiScanPacket)

HesaiScan = _reflection.GeneratedProtocolMessageType('HesaiScan', (_message.Message,), dict(
  DESCRIPTOR = _HESAISCAN,
  __module__ = 'modules.drivers.lidar.proto.hesai_pb2'
  # @@protoc_insertion_point(class_scope:apollo.drivers.hesai.HesaiScan)
  ))
_sym_db.RegisterMessage(HesaiScan)


# @@protoc_insertion_point(module_scope)
