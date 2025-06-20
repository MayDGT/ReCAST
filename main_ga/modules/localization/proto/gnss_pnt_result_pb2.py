# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modules/localization/proto/gnss_pnt_result.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

from  modules.drivers.gnss.proto import \
    gnss_raw_observation_pb2 as modules_dot_drivers_dot_gnss_dot_proto_dot_gnss__raw__observation__pb2

DESCRIPTOR = _descriptor.FileDescriptor(
  name='modules/localization/proto/gnss_pnt_result.proto',
  package='apollo.localization',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n0modules/localization/proto/gnss_pnt_result.proto\x12\x13\x61pollo.localization\x1a\x35modules/drivers/gnss/proto/gnss_raw_observation.proto\"f\n\x0cSatDirCosine\x12\x0f\n\x07sat_prn\x18\x01 \x01(\r\x12\x0f\n\x07sat_sys\x18\x02 \x01(\r\x12\x10\n\x08\x63osine_x\x18\x03 \x01(\x01\x12\x10\n\x08\x63osine_y\x18\x04 \x01(\x01\x12\x10\n\x08\x63osine_z\x18\x05 \x01(\x01\"\xac\x04\n\rGnssPntResult\x12\x13\n\x0breceiver_id\x18\x01 \x01(\r\x12>\n\ttime_type\x18\x02 \x01(\x0e\x32!.apollo.drivers.gnss.GnssTimeType:\x08GPS_TIME\x12\x11\n\tgnss_week\x18\x03 \x01(\r\x12\x15\n\rgnss_second_s\x18\x04 \x01(\x01\x12;\n\x08pnt_type\x18\x05 \x01(\x0e\x32\x1c.apollo.localization.PntType:\x0bPNT_INVALID\x12\x0f\n\x07pos_x_m\x18\x06 \x01(\x01\x12\x0f\n\x07pos_y_m\x18\x07 \x01(\x01\x12\x0f\n\x07pos_z_m\x18\x08 \x01(\x01\x12\x13\n\x0bstd_pos_x_m\x18\t \x01(\x01\x12\x13\n\x0bstd_pos_y_m\x18\n \x01(\x01\x12\x13\n\x0bstd_pos_z_m\x18\x0b \x01(\x01\x12\x0f\n\x07vel_x_m\x18\x0c \x01(\x01\x12\x0f\n\x07vel_y_m\x18\r \x01(\x01\x12\x0f\n\x07vel_z_m\x18\x0e \x01(\x01\x12\x13\n\x0bstd_vel_x_m\x18\x0f \x01(\x01\x12\x13\n\x0bstd_vel_y_m\x18\x10 \x01(\x01\x12\x13\n\x0bstd_vel_z_m\x18\x11 \x01(\x01\x12\x16\n\x0esovled_sat_num\x18\x12 \x01(\r\x12\x39\n\x0esat_dir_cosine\x18\x13 \x03(\x0b\x32!.apollo.localization.SatDirCosine\x12\x0c\n\x04pdop\x18\x14 \x01(\x01\x12\x0c\n\x04hdop\x18\x15 \x01(\x01\x12\x0c\n\x04vdop\x18\x16 \x01(\x01*r\n\x07PntType\x12\x0f\n\x0bPNT_INVALID\x10\x00\x12\x0b\n\x07PNT_SPP\x10\x01\x12\x10\n\x0cPNT_PHASE_TD\x10\x02\x12\x11\n\rPNT_CODE_DIFF\x10\x03\x12\x11\n\rPNT_RTK_FLOAT\x10\x04\x12\x11\n\rPNT_RTK_FIXED\x10\x05')
  ,
  dependencies=[modules_dot_drivers_dot_gnss_dot_proto_dot_gnss__raw__observation__pb2.DESCRIPTOR,])

_PNTTYPE = _descriptor.EnumDescriptor(
  name='PntType',
  full_name='apollo.localization.PntType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='PNT_INVALID', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PNT_SPP', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PNT_PHASE_TD', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PNT_CODE_DIFF', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PNT_RTK_FLOAT', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PNT_RTK_FIXED', index=5, number=5,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=791,
  serialized_end=905,
)
_sym_db.RegisterEnumDescriptor(_PNTTYPE)

PntType = enum_type_wrapper.EnumTypeWrapper(_PNTTYPE)
PNT_INVALID = 0
PNT_SPP = 1
PNT_PHASE_TD = 2
PNT_CODE_DIFF = 3
PNT_RTK_FLOAT = 4
PNT_RTK_FIXED = 5



_SATDIRCOSINE = _descriptor.Descriptor(
  name='SatDirCosine',
  full_name='apollo.localization.SatDirCosine',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sat_prn', full_name='apollo.localization.SatDirCosine.sat_prn', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sat_sys', full_name='apollo.localization.SatDirCosine.sat_sys', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cosine_x', full_name='apollo.localization.SatDirCosine.cosine_x', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cosine_y', full_name='apollo.localization.SatDirCosine.cosine_y', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cosine_z', full_name='apollo.localization.SatDirCosine.cosine_z', index=4,
      number=5, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=128,
  serialized_end=230,
)


_GNSSPNTRESULT = _descriptor.Descriptor(
  name='GnssPntResult',
  full_name='apollo.localization.GnssPntResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='receiver_id', full_name='apollo.localization.GnssPntResult.receiver_id', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='time_type', full_name='apollo.localization.GnssPntResult.time_type', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gnss_week', full_name='apollo.localization.GnssPntResult.gnss_week', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gnss_second_s', full_name='apollo.localization.GnssPntResult.gnss_second_s', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pnt_type', full_name='apollo.localization.GnssPntResult.pnt_type', index=4,
      number=5, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pos_x_m', full_name='apollo.localization.GnssPntResult.pos_x_m', index=5,
      number=6, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pos_y_m', full_name='apollo.localization.GnssPntResult.pos_y_m', index=6,
      number=7, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pos_z_m', full_name='apollo.localization.GnssPntResult.pos_z_m', index=7,
      number=8, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='std_pos_x_m', full_name='apollo.localization.GnssPntResult.std_pos_x_m', index=8,
      number=9, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='std_pos_y_m', full_name='apollo.localization.GnssPntResult.std_pos_y_m', index=9,
      number=10, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='std_pos_z_m', full_name='apollo.localization.GnssPntResult.std_pos_z_m', index=10,
      number=11, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='vel_x_m', full_name='apollo.localization.GnssPntResult.vel_x_m', index=11,
      number=12, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='vel_y_m', full_name='apollo.localization.GnssPntResult.vel_y_m', index=12,
      number=13, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='vel_z_m', full_name='apollo.localization.GnssPntResult.vel_z_m', index=13,
      number=14, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='std_vel_x_m', full_name='apollo.localization.GnssPntResult.std_vel_x_m', index=14,
      number=15, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='std_vel_y_m', full_name='apollo.localization.GnssPntResult.std_vel_y_m', index=15,
      number=16, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='std_vel_z_m', full_name='apollo.localization.GnssPntResult.std_vel_z_m', index=16,
      number=17, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sovled_sat_num', full_name='apollo.localization.GnssPntResult.sovled_sat_num', index=17,
      number=18, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sat_dir_cosine', full_name='apollo.localization.GnssPntResult.sat_dir_cosine', index=18,
      number=19, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pdop', full_name='apollo.localization.GnssPntResult.pdop', index=19,
      number=20, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hdop', full_name='apollo.localization.GnssPntResult.hdop', index=20,
      number=21, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='vdop', full_name='apollo.localization.GnssPntResult.vdop', index=21,
      number=22, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=233,
  serialized_end=789,
)

_GNSSPNTRESULT.fields_by_name['time_type'].enum_type = modules_dot_drivers_dot_gnss_dot_proto_dot_gnss__raw__observation__pb2._GNSSTIMETYPE
_GNSSPNTRESULT.fields_by_name['pnt_type'].enum_type = _PNTTYPE
_GNSSPNTRESULT.fields_by_name['sat_dir_cosine'].message_type = _SATDIRCOSINE
DESCRIPTOR.message_types_by_name['SatDirCosine'] = _SATDIRCOSINE
DESCRIPTOR.message_types_by_name['GnssPntResult'] = _GNSSPNTRESULT
DESCRIPTOR.enum_types_by_name['PntType'] = _PNTTYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SatDirCosine = _reflection.GeneratedProtocolMessageType('SatDirCosine', (_message.Message,), dict(
  DESCRIPTOR = _SATDIRCOSINE,
  __module__ = 'modules.localization.proto.gnss_pnt_result_pb2'
  # @@protoc_insertion_point(class_scope:apollo.localization.SatDirCosine)
  ))
_sym_db.RegisterMessage(SatDirCosine)

GnssPntResult = _reflection.GeneratedProtocolMessageType('GnssPntResult', (_message.Message,), dict(
  DESCRIPTOR = _GNSSPNTRESULT,
  __module__ = 'modules.localization.proto.gnss_pnt_result_pb2'
  # @@protoc_insertion_point(class_scope:apollo.localization.GnssPntResult)
  ))
_sym_db.RegisterMessage(GnssPntResult)


# @@protoc_insertion_point(module_scope)
