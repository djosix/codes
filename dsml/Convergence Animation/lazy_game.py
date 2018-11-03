import pygame

pg = pygame

class Game(object):
    def __init__(self,
                 name = "Frame",
                 size = (640, 480),
                 background = (255, 255, 255)):
        self.name = name
        self.size = size
        self.background = background
        self.logic_part = lambda: None
        self.clear_part = lambda: None
        self.draw_part = lambda: None
        pygame.init()
    
    def main(self):
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption(self.name)
 
        clock = pygame.time.Clock()
        
        self.done = False
        while not self.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
        
            self.logic_part()
            self.clear_part()
        
            if self.background:
                self.screen.fill(self.background)
        
            self.draw_part()
        
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()

    
    def logic(self, func):
        self.logic_part = func
    

    def clear(self, func):
        self.clear_part = func
    

    def draw(self, func):
        self.draw_part = func


class GameStart(Game):
    def __init__(self,
                 name = "Frame",
                 size = (640, 480),
                 background = (255, 255, 255),
                 logic = lambda g: None,
                 clear = lambda g: None,
                 draw = lambda g: None):
        self.name = name
        self.size = size
        self.background = background
        self.logic_part = logic
        self.clear_part = clear
        self.draw_part = draw
        self.main()

    def main(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption(self.name)
 
        clock = pygame.time.Clock()
        
        self.done = False
        while not self.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
        
            self.logic_part(self.screen)
            self.clear_part(self.screen)
        
            if self.background:
                self.screen.fill(self.background)
        
            self.draw_part(self.screen)
        
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
    