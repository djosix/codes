import os

with open('.playlist.m3u', 'w') as f:
    f.writelines(name for name in os.listdir() if name.endswith('.mp3'))

with open('.playlist.html', 'w') as f:
    f.write('''
    <embed name="playlist"
        src=".playlist.m3u"
        width="300"
        height="90"
        loop="false"
        hidden="false"
        autostart="true">
    </embed>
    '''


