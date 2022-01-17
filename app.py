from ctypes.wintypes import RGB
from datetime import date, datetime 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint
import copy
import matplotlib.animation as animation
from numpy.lib.function_base import average

g = 9.81
milliSecondDelayBetweenFrames = 20
stepsPerFrame = 32
deltaT = float(milliSecondDelayBetweenFrames) * 0.001 / float(stepsPerFrame)

m_1 = 0.1
l_1 = 0.3

m_2 = 0.1
l_2 = 0.3

# at t = 0
theta_1_0 = 2
theta_2_0 = 2
thetaDot_1_0 = 2
thetaDot_2_0 = 2

u_vector = [theta_1_0, theta_2_0, thetaDot_1_0, thetaDot_2_0]

u_vectorTimeSnapshots = []
def symplecticEulerOneStep():
    def findThetaDoubleDot_1(theta_1, theta_2, thetaDot_1, thetaDot_2):
        return (-m_2*l_1*thetaDot_1**2*np.sin(theta_1 - theta_2)*np.cos(theta_1 - theta_2) + m_2*g*np.sin(theta_2)*np.cos(theta_1 - theta_2) - m_2*l_2*thetaDot_2**2*np.sin(theta_1 - theta_2) - (m_1 + m_2)*g*np.sin(theta_1)) / ((m_1 + m_2)*l_1 - m_2*l_1*np.cos(theta_1 - theta_2)**2)

    def findThetaDoubleDot_2(theta_1, theta_2, thetaDot_1, thetaDot_2):
        return (m_2*l_2*thetaDot_2**2*np.sin(theta_1 - theta_2)*np.cos(theta_1 - theta_2) + (m_1 + m_2)*g*np.sin(theta_1)*np.cos(theta_1 - theta_2) + l_1*thetaDot_1**2*np.sin(theta_1-theta_2)*(m_1 + m_2) - g*np.sin(theta_2)*(m_1 + m_2)) / (l_1*(m_1 + m_2) - m_2*l_2*np.cos(theta_1 - theta_2)**2)

    u_vector[2] += findThetaDoubleDot_1(u_vector[0], u_vector[1], u_vector[2], u_vector[3]) * deltaT
    u_vector[3] += findThetaDoubleDot_2(u_vector[0], u_vector[1], u_vector[2], u_vector[3]) * deltaT

    u_vector[0] += u_vector[2] * deltaT
    u_vector[1] += u_vector[3] * deltaT

    u_vectorTimeSnapshots.append(copy.deepcopy(u_vector))

radiusOfGraphAxises = 0.5
backgroundColorRGB = (1, 1, 1)
lineColor = (0, 0, 0)

def getAxisCoordinatesOverTimeForParticle(_particle, _axis):
        line = []
        if _axis == 2:
            for i in range(len(u_vectorTimeSnapshots)):
                line.append(0)
            return line
        if _axis == 0 and _particle == 0:
            for i in range(len(u_vectorTimeSnapshots)):
                line.append(l_1*np.sin(u_vectorTimeSnapshots[i][0]))
            return line
        if _axis == 1 and _particle == 0:
            for i in range(len(u_vectorTimeSnapshots)):
                line.append(-l_1*np.cos(u_vectorTimeSnapshots[i][0]))
            return line
        if _axis == 0 and _particle == 1:
            for i in range(len(u_vectorTimeSnapshots)):
                line.append(l_1*np.sin(u_vectorTimeSnapshots[i][0]) + l_1*np.sin(u_vectorTimeSnapshots[i][1]))
            return line
        if _axis == 1 and _particle == 1:
            for i in range(len(u_vectorTimeSnapshots)):
                line.append(-l_1*np.cos(u_vectorTimeSnapshots[i][0]) - l_1*np.cos(u_vectorTimeSnapshots[i][1]))
            return line

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlim3d(-radiusOfGraphAxises, radiusOfGraphAxises)
ax.set_ylim3d(-radiusOfGraphAxises, radiusOfGraphAxises)
ax.set_zlim3d(-radiusOfGraphAxises + radiusOfGraphAxises*0.23, radiusOfGraphAxises - radiusOfGraphAxises*0.23)
ax.view_init(40, -5)
ax.set_title('Double Pendulum')
ax.set_facecolor(backgroundColorRGB)
fig.patch.set_facecolor(backgroundColorRGB)
plt.axis('off')

line0, = ax.plot(0, 0, 0, lw=0.5, color=((0, 0, 1)))
line1, = ax.plot(0, 0, 0, lw=0.5, color=((1, 0, 0)))
bar0, = ax.plot(0, 0, 0, lw=3, color=(lineColor))
bar1, = ax.plot(0, 0, 0, lw=3, color=(lineColor))

scats = []
starttime = datetime.now()
def animation_frame(i):  

    ax.view_init(0, -90)

    for x in range(stepsPerFrame):
        symplecticEulerOneStep()

    #averageFps = round(i / ((datetime.now() - starttime).total_seconds()))
    #ax.set_title(f'average fps: {averageFps}')
    ax.set_title('Double Pendulum')

    line0.set_data(getAxisCoordinatesOverTimeForParticle(0, 0), getAxisCoordinatesOverTimeForParticle(0, 2))
    line0.set_3d_properties(getAxisCoordinatesOverTimeForParticle(0, 1))

    line1.set_data(getAxisCoordinatesOverTimeForParticle(1, 0), getAxisCoordinatesOverTimeForParticle(1, 2))
    line1.set_3d_properties(getAxisCoordinatesOverTimeForParticle(1, 1))

    bar0.set_data([0, l_1*np.sin(u_vector[0])], [0, 0])
    bar0.set_3d_properties([0, -l_1*np.cos(u_vector[0])])

    bar1.set_data([l_1*np.sin(u_vector[0]), l_1*np.sin(u_vector[0]) + l_1*np.sin(u_vector[1])], [0, 0])
    bar1.set_3d_properties([-l_1*np.cos(u_vector[0]), -l_1*np.cos(u_vector[0]) - l_1*np.cos(u_vector[1])])

    global scats
    # first remove all old scatters
    for scat in scats:
        scat.remove()
    scats = []

    scats.append((ax.scatter(0, 0, 0, color=(0, 0, 0), s=3)))
    scats.append((ax.scatter(l_1*np.sin(u_vector[0]), 0, -l_1*np.cos(u_vector[0]), color=(0, 0, 1), s=10)))
    scats.append((ax.scatter(l_1*np.sin(u_vector[0]) + l_1*np.sin(u_vector[1]), 0, -l_1*np.cos(u_vector[0]) - l_1*np.cos(u_vector[1]), color=(1, 0, 0), s=10)))

ani = animation.FuncAnimation(fig, func=animation_frame, frames=np.arange(0, 750, 1), interval=milliSecondDelayBetweenFrames)
plt.show()

#ani.save('doublePendulum.gif')