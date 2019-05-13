import subprocess as sp
import json
import os, sys, tempfile, socket

class Sage:
    def __init__(self):
        self.sock_path = tempfile.mktemp()
        self.sage_proc = sp.Popen([
            'sage', '-c', f'''
def __main():
    import socket, json
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect({repr(self.sock_path)})
    sock_file = sock.makefile('rwb')
    while True:
        code = eval(sock_file.readline())
        result = __execute(code)
        result = json.dumps(result)
        sock_file.write(result + '\\n')
        sock_file.flush()

def __traceback():
    import traceback
    print traceback.format_exc()

def __execute(code):
    result = None
    try:
        exec(code)
    except Exception as e:
        result = None
        __traceback()
    return result

__main()
'''
        ])
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.sock_path)
        self.server.listen(1)
        self.client, _ = self.server.accept()
        self.client_file = self.client.makefile('rwb')

    def run(self, code):
        code = ('\n' + code).replace('\nreturn', 'result=')
        code_bytes = (repr(code) + '\n').encode()
        self.client_file.write(code_bytes)
        self.client_file.flush()
        result = self.client_file.readline().rstrip()
        result = json.loads(result)
        return result

    def close(self):
        self.server.close()
        self.sage_proc.kill()
        self.sage_proc.communicate()
        os.unlink(self.sock_path)

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

__all__ = ['Sage']



if __name__ == '__main__':
    sage = Sage()
    sage.run('print "this is from SageMath\'s stdout"')
    print(sage.run('''
return "this is returned from SageMath"
'''))
    print(sage.run('''
def fuck(name):
    return "fuck " + name

return fuck("you")
'''))
    sage.run('print factor(12837)')
    print(sage.run('return [1, 2, 3]') + [4, 5, 6])
    sage.close()
