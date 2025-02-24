from sudachipy import tokenizer
from sudachipy import dictionary
import sys
tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.B

for line in sys.stdin:
    line = line.strip()
    list = tokenizer_obj.tokenize(line, mode)
    print( [m.reading_form() for m in list] )

