from sheep_waver import SheepWaver

sw = SheepWaver('PTSND20011107.trs', 'PTSND20011107.WAV')
sw.cut_from_file_and_export('PTSND20011107_0.wav.trn')
print(sw.target)
print(sw.found)
