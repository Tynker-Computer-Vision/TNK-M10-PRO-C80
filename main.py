'''
SA1: Python file reading and Pickle
SA2: Removing unwanted code
SA3: Test the model on different track
'''
import pygame,math
import neat

import pickle
from helpers import getSensorX, getSensorY

config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,'config-feedforward.txt')  

# SA2 Load and store greatest fitness genome model
with open('std1.pkl', 'rb') as f:
    genome = pickle.load(f)
    

pygame.init()
screen = pygame.display.set_mode((800,600))

pygame.display.set_caption("Car racing")
background_image = pygame.image.load("track2.png").convert()
player_image = pygame.image.load("car.png").convert_alpha()

player=pygame.Rect(60,300,20,20)

WHITE=(255,255,255)
xvel=2
yvel=3
angle=0
change=0

# Creating a variable 'distance' and assigning value '5' to it
distance=2
# Creating a variable 'forward' and assigning 'False' to it
forward=False

font = pygame.font.Font('freesansbold.ttf', 12)

# Defining a function 'newxy()' to calculate new x,y coordinates
# New x,y coordinates are based on x,y coordinates, angle, distance 
def newxy(x,y,distance,angle):
  angle=math.radians(angle+90)
  xnew=x+(distance*math.cos(angle))
  # Calculating the new y-coordinate value 'ynew'
  ynew=y-(distance*math.sin(angle))
  # Returning 'xnew','ynew'
  return xnew,ynew

def checkOutOfBounds(car):
  x = car.x
  y = car.y
  width = car.width
  height = car.height

  if(checkPixel(x,y) or checkPixel(x+width, y) or checkPixel(x, y+height) or checkPixel(x+width, y+height)):
      return True
  
def checkPixel(x, y):
    global screen
    try:
        color = screen.get_at((x, y))
    except:
        return 1
    if(color == (163,171,160,255)):
        return 0
    return 1

def getSensorsData(car, angle):
    global screen
    margin = 55
    delta = 5
    x = car.x + car.w/2
    y = car.y + car.h/2
    
    sensorAngles = [-10,-30,-50,-70,-90,-110,-130,-150,-170]
    sensorData= []
    for sensorAngle in sensorAngles:
        sensorX = getSensorX(angle, sensorAngle)
        sensorY = getSensorY(angle, sensorAngle)

        newX = int(x -(margin * sensorX))
        newY = int(y +(margin * sensorY))
        
        
        sensorData.append(checkPixel(newX, newY))
         
        pygame.draw.rect(screen,(0, 255,0), [newX, newY, 5, 5])
        margin = margin + delta
        if margin > 70:
            delta = -delta
    
    print(sensorData)
    return sensorData[0], sensorData[1], sensorData[2], sensorData[3], sensorData[4], sensorData[5], sensorData[6], sensorData[7] 
    
angle =0
       
net = neat.nn.FeedForwardNetwork.create(genome, config)

while True:
  screen.blit(background_image,[0,0])
 
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      pygame.quit()
      
    if event.type == pygame.KEYDOWN:
       if event.key == pygame.K_LEFT:
          change = 5
       if event.key ==pygame.K_RIGHT:
        change = -5 

       if event.key == pygame.K_UP:
        forward = True
        
    if event.type == pygame.KEYUP:
      if event.key ==pygame.K_LEFT or event.key == pygame.K_RIGHT:
          change = 0
     
      if event.key == pygame.K_UP:
        forward = False 
    
 
  if forward:
      player.x,player.y=newxy(player.x, player.y, 3, angle)  
                  
  if(checkOutOfBounds(player)):
      player.x = 60
      player.y = 300
      angle =0
      break
  
  angle = angle + change
  
  newimage=pygame.transform.rotate(player_image,angle) 
  pygame.draw.rect(screen,(0, 255, 0), player)
  screen.blit(newimage ,player)
    
  # Controlling the game using neural network
  forward = True
  change = 0
  
  sensor1, sensor2, sensor3, sensor4, sensor5, sensor6, sensor7, sensor8 = getSensorsData(player, angle)
  output = net.activate((angle, sensor1, sensor2, sensor3, sensor4, sensor5, sensor6, sensor7, sensor8))
  
  
  inputText = font.render('All Sensors:'+ str(sensor1)+ str(sensor2)+ str(sensor3)+ str(sensor4)+ str(sensor5)+ str(sensor6)+ str(sensor7)+ str(sensor8) , True, (255,255,0))
  screen.blit(inputText, (420, 60))
  
  output1Text = font.render('Output1:'+ str(output[0]), True, (255,255,0))
  screen.blit(output1Text, (420, 80))
  output2Text = font.render('Output2:'+ str(output[1]), True, (255,255,0))
  screen.blit(output2Text, (420, 100))
  
  if output[0] > 0.65:
      change = 3
  if output[1] > 0.65:
      change = -3
      
  
  pygame.display.update()
  pygame.time.Clock().tick(30)
  
