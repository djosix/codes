''' Notes
[Reload Modules]
%load_ext autoreload
%autoreload 2
'''


v_gl = 'global variable'


def py():
    v_py = 'local variable in py'

    from code import interact
    scope = {}
    scope.update(globals())
    scope.update(locals())
    interact(local=scope)


def ipy():
    v_ipy = 'local variable in ipy'

    from IPython import embed as debug
    debug()


py()

ipy()
