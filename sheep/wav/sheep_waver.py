#!/usr/bin/env python3

from bs4 import BeautifulSoup
from pydub import AudioSegment
from difflib import SequenceMatcher
from string import printable
from collections import namedtuple


USAGE = '''
~~~~~~~~~~~~~~~
~~ SheepWaver ~~
~~~~~~~~~~~~~~~

Usage: python3 {} XML AUDIO TARGET_TEXT EXPORT_PATH
'''


''' NOTE

!!! PYTHON 3 !!!


source_wav = 'PTSND20011107.WAV'
source_text = 'PTSND20011107.trs'
target_text = 'PTSND20011107_0.wav.trn'
target_wav = 'PTSND20011107_0.wav'

SequenceMatcher().ratio()

sw = SheepWaver('PTSND20011107.trs', 'PTSND20011107.WAV')
'''


Segment = namedtuple('Segment', ('begin', 'end', 'text'))
Result = namedtuple('Result', ('segments', 'text'))


class SheepWaver:
    def __init__(self, xml, audio):
        '''
        Args:
            xml: path of source xml file
            audio: path of source audio file
        
        Usage:
            >>> from sheep_waver import SheepWaver
            >>> sw = SheepWaver('PTSND20011107.trs', 'PTSND20011107.WAV')
            >>> sw.cut_from_file_and_export('PTSND20011107_0.wav.trn')
            >>> print(sw.target)
            >>> print(sw.found)
        '''
        with open(xml, 'rb') as f:
            self.xml = BeautifulSoup(f.read(), 'lxml')
        with open(audio, 'rb') as f:
            self.audio = AudioSegment(f)
        self._parse_xml()
    
    def cut(self, text):
        '''
        Args:
            text: text to find

        Returns:
            AudioSegment object
        
        Usage:
            >>> sw.cut('大肥羊').export('sheep.wav')
        '''
        t0, t1 = self._get_period(text)
        return self.audio[t0*1000:t1*1000]
    
    def cut_from_file_and_export(self, text_path, export_path='output.wav'):
        '''
        Args:
            text_path:   path of the text file
            export_path: path to export the audio
        
        Usage:
            >>> sw = SheepWaver('PTSND20011107.trs', 'PTSND20011107.WAV')
            >>> sw.cut_from_file_and_export('PTSND20011107_0.wav.trn', 'myOutput.wav')
        '''
        try:
            with open(text_path, 'rt') as f:
                text = f.read()
        except:
            # for the fucking big5 of the fucking Windows users
            with open(text_path, 'rb') as f:
                text = f.read().decode('cp950')
        self.cut(text).export(export_path)

    def _parse_xml(self):
        syncs = self.xml.find_all('sync')
        junk = {'，', '。', '？', '！', '：', '；', '「', '」', '『', '』'} | set(printable)
        segments = []
        for sync in syncs:
            text = ''
            for sib in sync.next_siblings:
                if sib.name == None:
                    text += str(sib)
                if sib in syncs:
                    text = ''.join([c for c in text if c not in junk])
                    if text:
                        segment = Segment(float(sync.get('time')),
                                          float(sib.get('time')),
                                          text)
                        segments.append(segment)
                    break
        self.segs = segments

    def _get_period(self, text):
        self.target = text
        argmax, ratio = -1, 0
        for i, s in enumerate(self.segs):
            r = SequenceMatcher(None, s.text, text).ratio()
            if r > ratio:
                argmax, ratio = i, r
        si = [argmax] # final segments indexes
        main = self.segs[argmax].text # main string of segments to find
        for i in reversed(range(0, si[0])):
            temp = self.segs[i].text + main
            r = SequenceMatcher(None, temp, text).ratio()
            if r < ratio:
                break
            main, ratio, si = temp, r, [i] + si
        for i in range(si[-1] + 1, len(self.segs)):
            temp = main + self.segs[i].text
            r = SequenceMatcher(None, temp, text).ratio()
            if r < ratio:
                break
            main, ratio, si = temp, r, si + [i]
        self.found = Result(si, main)
        return self.segs[si[0]].begin, self.segs[si[-1]].end


if __name__ == '__main__':
    import sys, os
    if len(sys.argv) != 5:
        print(USAGE.format(os.path.basename(sys.argv[0])))
    else:
        sw = SheepWaver(*sys.argv[1:3])
        sw.cut_from_file_and_export(*sys.argv[3:5])
