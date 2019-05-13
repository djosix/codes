import subprocess as sp
import os, sys
import tempfile
import socket
import pickle
import base64


def deindent(code):
    import re
    try:
        return code.replace(re.search(r'\n +', code).group(), '\n')
    except:
        return code


class Sage:
    def __init__(self, stderr=True):
        self.sock_path = tempfile.mktemp()
        sage_script = deindent(f'''
        def __main():
            import socket, pickle, base64
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect({repr(self.sock_path)})
            sock_file = sock.makefile('rwb')
            while True:
                code = eval(sock_file.readline())
                try:
                    result = __execute(code)
                    result = pickle.dumps(result)
                    result = base64.b64encode(result)
                except:
                    import traceback
                    print traceback.format_exc()
                    result = base64.b64encode(pickle.dumps(None))
                sock_file.write(result + '\\n')
                sock_file.flush()

        def __execute(code):
            __return_value = None
            exec(code)
            return __return_value

        __main()
        ''')
        self.sage_proc = sp.Popen(['sage', '-c', sage_script],
                                  stderr=(sp.STDOUT if stderr else sp.DEVNULL))

        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.sock_path)
        self.server.listen(1)

        self.client, _ = self.server.accept()
        self.client_file = self.client.makefile('rwb')

    def run(self, code, indent=True):
        if indent:
            code = deindent(code)
        code = ('\n' + code).replace('\nreturn', '__return_value=')
        code_bytes = (repr(code) + '\n').encode()

        self.client_file.write(code_bytes)
        self.client_file.flush()

        result = self.client_file.readline()
        result = base64.b64decode(result)
        result = pickle.loads(result, encoding='bytes')
        return result

    def close(self):
        self.client.close()
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

    with Sage() as sage:
        # you can run code in sage
        sage.run('print "This is from SageMath stdout"')

        # you can return something to python3
        # strings in python2 are bytes
        print(sage.run('return "This is returned from SageMath Python"'))

        # unicode string is default in python3
        print(sage.run('return u"This is a unicode string"'))

        # multiline code is supported
        print(sage.run('''
        def fuck(name):
            return "fuck " + name

        return fuck("you")
        '''))

        # sage functions are available
        sage.run('print factor(12837)')

        # this type cannot be pickled, so there will be a None returned
        print(sage.run('return factor(29229)'))

        # returning complicated objects
        print(sage.run('''
        return {
            'this': 'can be transfer in pickle format',
            'and so does this': [
                0, 1, 2, 3, None, Exception('FUCK'),
            ],
            'set': set([1,2,3,4,2,1,2]),
            'tuple': tuple(),
        }
        '''))
