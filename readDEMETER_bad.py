"""
import codecs
myfile = '/home/rileyannereid/workspace/canvas/DEMETER_data/ORDER_lairboulder_26539/DA_TC_DMT_N1_1131/DMT_N1_1131_000420_20040705_080230_20040705_080600.DAT'

with open(myfile, "rb") as file:
     data = file.read(100)

file.close()

with open("out.txt", "w") as f:
   f.write(" ".join(map(str,data)))
   f.write("\n")
"""

# dont do this lol ^^^

import idlsave

idlfile = '/home/rileyannereid/workspace/canvas/DEMETER_data/IDL/rd_dmt_n1.sav'
s = idlsave.read(idlfile)