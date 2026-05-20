import sys
import json
from importlib import resources


def _load_single_char_dic() -> dict:
    data = (resources.files("momo_py") / "single_character_dic.json").read_bytes()
    return json.loads(data.decode("utf-8"))


def lookup(c: str) -> None:
    dic = _load_single_char_dic()
    readings = dic.get(c)
    if not readings:
        print(f"No readings found for '{c}'")
        return
    print(f"Readings for '{c}':")
    for reading in readings:
        print(f"  {reading}")


def main():
    if len(sys.argv) < 2:
        print("Usage: lookup <char>")
        return
    lookup(sys.argv[1])


if __name__ == "__main__":
    main()
