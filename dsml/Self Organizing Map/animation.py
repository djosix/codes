import pygame


class App:
    def __init__(self,
                 name='App',
                 size=(640, 480),
                 background=(255, 255, 255),
                 fps=60):
        self.name = name
        self.size = size
        self.background = background
        self.fps = fps
        self.done = False
        self.screen = None
        self.clock = None
        pygame.init()
    

    def start(self):

        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption(self.name)
 
        self.clock = pygame.time.Clock()
        
        while not self.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
        
            self.run('before_logic')
            self.run('logic')
            self.run('after_logic')
        
            if self.background:
                self.screen.fill(self.background)
        
            self.run('before_render')
            self.run('render')
            self.run('after_render')

            pygame.display.flip()
            
            if self.fps:
                self.clock.tick(self.fps)
        
        pygame.quit()


    def use(self, func):
        setattr(self, func.__name__, func)
        return func


    def run(self, func):
        if hasattr(self, func):
            getattr(self, func)()



if __name__ == '__main__':
    app = App(name='Testing', size=(800, 600))
    Color, draw = pygame.Color, pygame.draw

    import numpy as np
    mid = np.array(app.size, dtype=float) / 2
    pos = np.zeros(2)
    v = np.array([50., 0.])

    @app.use
    def logic():
        global mid, pos, v
        v += (mid - pos) / 50
        v *= 0.99
        pos += v

    @app.use
    def render():
        draw.circle(app.screen, Color('black'), pos.astype(int), 10)
        draw.line(app.screen, Color('black'), pos, (0, 0))
        draw.line(app.screen, Color('black'), pos, (app.size[0], 0))
        draw.line(app.screen, Color('black'), pos, (0, app.size[1]))
        draw.line(app.screen, Color('black'), pos, app.size)

    app.start()
