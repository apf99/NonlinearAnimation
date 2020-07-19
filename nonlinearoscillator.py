import pygame
import sys
from pygame.locals import *
from math import sin, cos, tan, pi
import numpy as np
from numpy.linalg import inv, norm
from spring import spring

class Spring():
	def __init__(self, color, start, end, nodes, width, lead1, lead2):
		self.start = start
		self.end = end
		self.nodes = nodes
		self.width = width
		self.lead1 = lead1
		self.lead2 = lead2
		self.weight = 3
		self.color = color

	def update(self, start, end):
		self.start = start
		self.end = end

		self.x, self.y, self.p1, self.p2 = spring(self.start, self.end, self.nodes, self.width, self.lead1, self.lead2)
		self.p1 = (int(self.p1[0]), int(self.p1[1]))
		self.p2 = (int(self.p2[0]), int(self.p2[1]))

	def render(self):
		pygame.draw.line(screen, self.color, self.start, self.p1, self.weight)
		prev_point = self.p1
		for point in zip(self.x, self.y):
			pygame.draw.line(screen, self.color, prev_point, point, self.weight)
			prev_point = point
		pygame.draw.line(screen, self.color, self.p2, self.end, self.weight)
	


def G2(y,t): 
	x1_d, x2_d, x1, x2 = y[0], y[1], y[2], y[3]

	r1 = np.array([l1+x1, x2])
	r2 = np.array([x1, l2+x2])

	A1 = (1 - l1/norm(r1))
	A2 = (1 - l2/norm(r2))

	x1_dd =  (-A1*k1*(x1+l1) - A2*k2*x1) / m
	x2_dd =  (-A1*k1*x2 - A2*k2*(x2+l2)) / m

	return np.array([x1_dd, x2_dd, x1_d, x2_d])


def G(y,t):
	x1_d, x2_d, x1, x2 = y[0], y[1], y[2], y[3]

	e1 = np.array([1,0])
	e2 = np.array([0,1])

	r1 = np.array([l1+x1, x2])
	r2 = np.array([x1, l2+x2])

	er1 = r1 / norm(r1)
	er2 = r2 / norm(r2)

	delta1 = norm(r1) - l1
	delta2 = norm(r2) - l2

	F1 = -k1 * delta1 * er1
	F2 = -k2 * delta2 * er2

	x1_dd = np.dot((F1+F2), e1) / m
	x2_dd = np.dot((F1+F2), e2) / m

	return np.array([x1_dd, x2_dd, x1_d, x2_d])


def RK4_step(y, t, dt):
	k1 = G(y,t)
	k2 = G(y+0.5*k1*dt, t+0.5*dt)
	k3 = G(y+0.5*k2*dt, t+0.5*dt)
	k4 = G(y+k3*dt, t+dt)

	return dt * (k1 + 2*k2 + 2*k3 + k4) / 6

def update(x1, x2):
	x_coord = scale*x1 + offset[0]
	y_coord = -scale*x2 + offset[1]

	return (int(x_coord), int(y_coord))

def render(point):

	if prev_point:
		pygame.draw.line(trace, LT_BLUE, prev_point, point, 5)

	screen.fill(WHITE)	
	if is_tracing:
		screen.blit(trace, (0,0))


	p1 = (int(-l1*scale+offset[0]), int(offset[1]))
	p2 = (int(offset[0]), int(l2*scale+offset[1]))

	pygame.draw.line(trace, YELLOW, p1, p2, 2)

	s1.update(p1, point)
	s2.update(p2, point)
	s1.render()
	s2.render()

	pygame.draw.circle(screen, RED, offset, 2)

	offset2 = (int(-l1*scale + offset[0]), int(l2*scale + offset[1]))
	pygame.draw.circle(screen, RED, offset2, 2)

	pygame.draw.circle(screen, BLACK, p1, 8)
	pygame.draw.circle(screen, BLACK, p2, 8)
	pygame.draw.circle(screen, BLUE, point, int(m*5))

	return point

w, h = 1024, 768
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
BLUE = (0,0,255)
YELLOW = (240,240,0)
LT_BLUE = (230,230,255)
offset = (3*w//5, 2*h//5)
scale = 250
is_tracing = True

screen = pygame.display.set_mode((w,h))
screen.fill(WHITE)
trace = screen.copy()
pygame.display.update()
clock = pygame.time.Clock()

# parameters
m = 8.0
l1 = 1.0
l2 = 1.0
g = 9.81
k1 = 10.0
k2 = 10.0

prev_point = None
t = 0.0
delta_t = 0.05
y = np.array([0.0, 0.0, 0.7, 0.3]) 

pygame.font.init()
myfont = pygame.font.SysFont('Comic Sans MS', 38)

s1 = Spring(BLACK, (0,0), (0,0), 10, scale//4, scale//10, 70)
s2 = Spring(BLACK, (0,0), (0,0), 10, scale//4, scale//10, 70)

while True:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			sys.exit()

		if event.type == KEYDOWN:
			if event.key == K_t:
				is_tracing = not(is_tracing)
			if event.key == K_c:
				trace.fill(WHITE)

	point = update(y[2], y[3])
	prev_point = render(point)

	time_string = 'Time: {} seconds'.format(round(t,1))
	text = myfont.render(time_string, False, (0, 0, 0))
	screen.blit(text, (10,10))

	t += delta_t
	y = y + RK4_step(y, t, delta_t) 

	clock.tick(60)
	pygame.display.update()
