import braille_rules_pb2
from google.protobuf import text_format
import sys
braille_rules = braille_rules_pb2.BrailleRules()

def CreateBrailleRules():
    rules = braille_rules_pb2.BrailleRules()
    # '名詞'
    rule = rules.rule.add()
    rule.current_pos.name = '名詞'
    n_pos = rule.next_pos.add()
    n_pos.name = '助詞'
    n_pos.before_space = False
    return rules

def WriteBrailleRules():
    with open('backup.textproto', 'w') as f:
        f.write(text_format.MessageToString(braille_rules))

def LoadBrailleRules():
    try:
        with open('braille_rules.textproto', 'r') as f:
            text_format.Parse(f.read(), braille_rules)
    except FileNotFoundError:
        print("Error: The file 'braille_rules.textproto' was not found.", file=sys.stderr)
    except IOError as e:
        print(f"Error: An I/O error occurred: {e}", file=sys.stderr)

def ProcessBrailleRules(inString):
    return inString

if __name__ == '__main__':
    LoadBrailleRules()
    WriteBrailleRules()
