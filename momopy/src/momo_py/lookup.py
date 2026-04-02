import sys
from enum import Enum
from momo_py.features import CharType
from momo_py.translator import Translator

def lookup(c: str):
    t = Translator()
    readings = t._get_reading_form_in_dictionary(c)
    if not readings:
        print(f"No readings found for '{c}'")
        return
    print(f"Readings for '{c}':")
    for reading in readings:
        print(f"  {reading}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python lookup.py <text>")
        return
    text = sys.argv[1]
    lookup(text)

if __name__ == "__main__":
    main()
