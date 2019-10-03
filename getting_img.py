import pygame

class get_img(object):
	
	pygame.init()
	def __init__(self):
		self.winx, self.winy = 100, 100
		#The drawing cursor width 
		self.cursor_width = 5
		self.win = pygame.display.set_mode((self.winx,self.winy))
	def get(self):
		run = True
		while run :
			self.win.fill((0,0,0))
			pygame.time.delay(10)
			for event in pygame.event.get() :
				if event.type == pygame.QUIT:
					run = False
				if event.type == pygame.MOUSEBUTTONDOWN:
					get_img = True 
					while get_img :
						for event in pygame.event.get() :
							if event.type == pygame.MOUSEBUTTONUP:
								get_img = False 
						x,y = pygame.mouse.get_pos()
						pygame.draw.circle(self.win,(255,255,255),(x,y),self.cursor_width)
						pygame.display.update()
					image_data = pygame.surfarray.array3d(pygame.display.get_surface())
					return image_data
			pygame.display.update()


