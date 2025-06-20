# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modules/control/proto/lon_controller_conf.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

from  modules.control.proto import pid_conf_pb2 as modules_dot_control_dot_proto_dot_pid__conf__pb2, \
    leadlag_conf_pb2 as modules_dot_control_dot_proto_dot_leadlag__conf__pb2, \
    calibration_table_pb2 as modules_dot_control_dot_proto_dot_calibration__table__pb2

DESCRIPTOR = _descriptor.FileDescriptor(
  name='modules/control/proto/lon_controller_conf.proto',
  package='apollo.control',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n/modules/control/proto/lon_controller_conf.proto\x12\x0e\x61pollo.control\x1a-modules/control/proto/calibration_table.proto\x1a(modules/control/proto/leadlag_conf.proto\x1a$modules/control/proto/pid_conf.proto\"!\n\nFilterConf\x12\x13\n\x0b\x63utoff_freq\x18\x01 \x01(\x05\"\xec\x06\n\x11LonControllerConf\x12\n\n\x02ts\x18\x01 \x01(\x01\x12\x1c\n\x14\x62rake_minimum_action\x18\x02 \x01(\x01\x12\x1f\n\x17throttle_minimum_action\x18\x03 \x01(\x01\x12$\n\x1cspeed_controller_input_limit\x18\x04 \x01(\x01\x12\x1b\n\x13station_error_limit\x18\x05 \x01(\x01\x12\x16\n\x0epreview_window\x18\x06 \x01(\x01\x12\x1f\n\x17standstill_acceleration\x18\x07 \x01(\x01\x12\x31\n\x10station_pid_conf\x18\x08 \x01(\x0b\x32\x17.apollo.control.PidConf\x12\x33\n\x12low_speed_pid_conf\x18\t \x01(\x0b\x32\x17.apollo.control.PidConf\x12\x34\n\x13high_speed_pid_conf\x18\n \x01(\x0b\x32\x17.apollo.control.PidConf\x12\x14\n\x0cswitch_speed\x18\x0b \x01(\x01\x12\x39\n\x18reverse_station_pid_conf\x18\x0c \x01(\x0b\x32\x17.apollo.control.PidConf\x12\x37\n\x16reverse_speed_pid_conf\x18\r \x01(\x0b\x32\x17.apollo.control.PidConf\x12;\n\x17pitch_angle_filter_conf\x18\x0e \x01(\x0b\x32\x1a.apollo.control.FilterConf\x12\x41\n\x1creverse_station_leadlag_conf\x18\x0f \x01(\x0b\x32\x1b.apollo.control.LeadlagConf\x12?\n\x1areverse_speed_leadlag_conf\x18\x10 \x01(\x0b\x32\x1b.apollo.control.LeadlagConf\x12S\n\x11\x63\x61libration_table\x18\x11 \x01(\x0b\x32\x38.apollo.control.calibrationtable.ControlCalibrationTable\x12\x32\n#enable_reverse_leadlag_compensation\x18\x12 \x01(\x08:\x05\x66\x61lse\x12\x1e\n\x13switch_speed_window\x18\x13 \x01(\x01:\x01\x30')
  ,
  dependencies=[modules_dot_control_dot_proto_dot_calibration__table__pb2.DESCRIPTOR,modules_dot_control_dot_proto_dot_leadlag__conf__pb2.DESCRIPTOR,modules_dot_control_dot_proto_dot_pid__conf__pb2.DESCRIPTOR,])




_FILTERCONF = _descriptor.Descriptor(
  name='FilterConf',
  full_name='apollo.control.FilterConf',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='cutoff_freq', full_name='apollo.control.FilterConf.cutoff_freq', index=0,
      number=1, type=5, cpp_type=1, label=1,
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
  serialized_start=194,
  serialized_end=227,
)


_LONCONTROLLERCONF = _descriptor.Descriptor(
  name='LonControllerConf',
  full_name='apollo.control.LonControllerConf',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ts', full_name='apollo.control.LonControllerConf.ts', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='brake_minimum_action', full_name='apollo.control.LonControllerConf.brake_minimum_action', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='throttle_minimum_action', full_name='apollo.control.LonControllerConf.throttle_minimum_action', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='speed_controller_input_limit', full_name='apollo.control.LonControllerConf.speed_controller_input_limit', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='station_error_limit', full_name='apollo.control.LonControllerConf.station_error_limit', index=4,
      number=5, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='preview_window', full_name='apollo.control.LonControllerConf.preview_window', index=5,
      number=6, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='standstill_acceleration', full_name='apollo.control.LonControllerConf.standstill_acceleration', index=6,
      number=7, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='station_pid_conf', full_name='apollo.control.LonControllerConf.station_pid_conf', index=7,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='low_speed_pid_conf', full_name='apollo.control.LonControllerConf.low_speed_pid_conf', index=8,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='high_speed_pid_conf', full_name='apollo.control.LonControllerConf.high_speed_pid_conf', index=9,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='switch_speed', full_name='apollo.control.LonControllerConf.switch_speed', index=10,
      number=11, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reverse_station_pid_conf', full_name='apollo.control.LonControllerConf.reverse_station_pid_conf', index=11,
      number=12, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reverse_speed_pid_conf', full_name='apollo.control.LonControllerConf.reverse_speed_pid_conf', index=12,
      number=13, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pitch_angle_filter_conf', full_name='apollo.control.LonControllerConf.pitch_angle_filter_conf', index=13,
      number=14, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reverse_station_leadlag_conf', full_name='apollo.control.LonControllerConf.reverse_station_leadlag_conf', index=14,
      number=15, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reverse_speed_leadlag_conf', full_name='apollo.control.LonControllerConf.reverse_speed_leadlag_conf', index=15,
      number=16, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='calibration_table', full_name='apollo.control.LonControllerConf.calibration_table', index=16,
      number=17, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='enable_reverse_leadlag_compensation', full_name='apollo.control.LonControllerConf.enable_reverse_leadlag_compensation', index=17,
      number=18, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='switch_speed_window', full_name='apollo.control.LonControllerConf.switch_speed_window', index=18,
      number=19, type=1, cpp_type=5, label=1,
      has_default_value=True, default_value=float(0),
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
  serialized_start=230,
  serialized_end=1106,
)

_LONCONTROLLERCONF.fields_by_name['station_pid_conf'].message_type = modules_dot_control_dot_proto_dot_pid__conf__pb2._PIDCONF
_LONCONTROLLERCONF.fields_by_name['low_speed_pid_conf'].message_type = modules_dot_control_dot_proto_dot_pid__conf__pb2._PIDCONF
_LONCONTROLLERCONF.fields_by_name['high_speed_pid_conf'].message_type = modules_dot_control_dot_proto_dot_pid__conf__pb2._PIDCONF
_LONCONTROLLERCONF.fields_by_name['reverse_station_pid_conf'].message_type = modules_dot_control_dot_proto_dot_pid__conf__pb2._PIDCONF
_LONCONTROLLERCONF.fields_by_name['reverse_speed_pid_conf'].message_type = modules_dot_control_dot_proto_dot_pid__conf__pb2._PIDCONF
_LONCONTROLLERCONF.fields_by_name['pitch_angle_filter_conf'].message_type = _FILTERCONF
_LONCONTROLLERCONF.fields_by_name['reverse_station_leadlag_conf'].message_type = modules_dot_control_dot_proto_dot_leadlag__conf__pb2._LEADLAGCONF
_LONCONTROLLERCONF.fields_by_name['reverse_speed_leadlag_conf'].message_type = modules_dot_control_dot_proto_dot_leadlag__conf__pb2._LEADLAGCONF
_LONCONTROLLERCONF.fields_by_name['calibration_table'].message_type = modules_dot_control_dot_proto_dot_calibration__table__pb2._CONTROLCALIBRATIONTABLE
DESCRIPTOR.message_types_by_name['FilterConf'] = _FILTERCONF
DESCRIPTOR.message_types_by_name['LonControllerConf'] = _LONCONTROLLERCONF
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

FilterConf = _reflection.GeneratedProtocolMessageType('FilterConf', (_message.Message,), dict(
  DESCRIPTOR = _FILTERCONF,
  __module__ = 'modules.control.proto.lon_controller_conf_pb2'
  # @@protoc_insertion_point(class_scope:apollo.control.FilterConf)
  ))
_sym_db.RegisterMessage(FilterConf)

LonControllerConf = _reflection.GeneratedProtocolMessageType('LonControllerConf', (_message.Message,), dict(
  DESCRIPTOR = _LONCONTROLLERCONF,
  __module__ = 'modules.control.proto.lon_controller_conf_pb2'
  # @@protoc_insertion_point(class_scope:apollo.control.LonControllerConf)
  ))
_sym_db.RegisterMessage(LonControllerConf)


# @@protoc_insertion_point(module_scope)
