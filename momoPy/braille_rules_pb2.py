# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: protos/braille_rules.proto
# Protobuf Python Version: 5.29.3
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    3,
    '',
    'protos/braille_rules.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1aprotos/braille_rules.proto\x12\x0e\x62raillel_rules\"\xd4\x01\n\x0cPartOfSpeech\x12\x0c\n\x04name\x18\x01 \x03(\t\x12\x14\n\x0c\x62\x65\x66ore_space\x18\x02 \x01(\x08\x12\x13\n\x0b\x61\x66ter_space\x18\x03 \x01(\x08\x12\x18\n\x10\x61llow_line_break\x18\x04 \x01(\x08\x12\x12\n\nword_match\x18\x05 \x03(\t\x12\x16\n\x0eword_not_match\x18\x06 \x03(\t\x12 \n\x18reading_word_length_less\x18\x07 \x01(\x05\x12#\n\x1breading_word_length_greater\x18\x08 \x01(\x05\"i\n\x04Rule\x12\x31\n\x0b\x63urrent_pos\x18\x01 \x01(\x0b\x32\x1c.braillel_rules.PartOfSpeech\x12.\n\x08next_pos\x18\x02 \x03(\x0b\x32\x1c.braillel_rules.PartOfSpeech\"2\n\x0c\x42railleRules\x12\"\n\x04rule\x18\x01 \x03(\x0b\x32\x14.braillel_rules.Ruleb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'protos.braille_rules_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_PARTOFSPEECH']._serialized_start=47
  _globals['_PARTOFSPEECH']._serialized_end=259
  _globals['_RULE']._serialized_start=261
  _globals['_RULE']._serialized_end=366
  _globals['_BRAILLERULES']._serialized_start=368
  _globals['_BRAILLERULES']._serialized_end=418
# @@protoc_insertion_point(module_scope)
