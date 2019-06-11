import pygame as pg

winx, winy = 1200,700
pg.init()
win = pg.display.set_mode((winx,winy))


#====================================================================#
# Details about the layers. #
#====================================================================#
maxLayers, minLayers = 20, 0
noOfLayers, nodesInLayers = 2, []
input_layer, output_layer = 784, 10
for i in range(noOfLayers) :
	nodesInLayers.append(64)

#====================================================================#

#====================================================================#
# The stack functions which will draw a layer. #
#====================================================================#
def drawLayer(nodes, winx, winy, x):
	max_radius, min_radius = 20, 2
	calc_radius = winy//(3*nodes)
	if calc_radius < min_radius :
		calc_radius = min_radius
	if calc_radius > max_radius :
		calc_radius = max_radius
	cury = 3*calc_radius
	for i in range(nodes) :
		pg.draw.circle(win,(0,0,0),(100, cury), calc_radius)
		cury += 3*calc_radius


#====================================================================#

def redraw():
	win.fill((255,255,255))
	drawLayer(115,winx,winy,100)
	pg.display.update()

run = True 
while run :
	pg.time.delay(10)
	redraw()
	for event in pg.event.get() :
		if event.type == pg.QUIT:
			run =False 
