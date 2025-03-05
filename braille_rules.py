import braille_rules_pb2
from google.protobuf import text_format
import sys
rules = braille_rules_pb2.BrailleRules()

def create_braille_rules():
    rules = braille_rules_pb2.BrailleRules()
    # '名詞'
    rule = rules.rule.add()
    rule.current_pos.name = '名詞'
    n_pos = rule.next_pos.add()
    n_pos.name = '助詞'
    n_pos.before_space = False
    return rules

def write_braille_rules():
    with open('backup.textproto', 'w') as f:
        f.write(text_format.MessageToString(rules))

def load_b_railleRules():
    try:
        with open('braille_rules.textproto', 'r') as f:
            text_format.Parse(f.read(), rules)
    except FileNotFoundError:
        print("Error: The file 'braille_rules.textproto' was not found.", file=sys.stderr)
    except IOError as e:
        print(f"Error: An I/O error occurred: {e}", file=sys.stderr)

if __name__ == '__main__':
    load_b_railleRules()
    WriteBrailleRules()
