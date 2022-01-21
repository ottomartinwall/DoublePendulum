from ctypes.wintypes import RGB
from datetime import date, datetime
from typing import Counter 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint
import copy
import matplotlib.animation as animation
from numpy.lib.function_base import average
from matplotlib.backend_bases import MouseButton

#-INFO-----------------------------------------------------------------------------------------------------------------------------

# Install the necessary packages and then run this code. 
# The window that pops up is interactable so you can click somewhere to drop the pendulum from there.

#-CONTROLLING VARIABLES -----------------------------------------------------------------------------------------------------------

fps = 60 # choose a stable framerate for 1:1 time in simulation
stepsPerFrame = 32 # increase for improved accuracy but lesser performance
lineTime = 0.25 # amount of seconds that colored lines show behind particles

m_1 = 0.8 # mass of pendulum 1 (kg)
l_1 = 0.5 # length of pendulum 1 (m)

m_2 = 0.8 # mass of pendulum 2 (kg)
l_2 = 0.5 # length of pendulum 2 (m)

g = 9.81 # gravity (m/s)
frictionCoefficient = 0.1 # (dimensionless)

theta_1_0 = 2 # starting angle for pendulum 1
theta_2_0 = 2 # starting angle for pendulum 2
thetaDot_1_0 = 0 # starting angular velocity for pendulum 1
thetaDot_2_0 = 0 # starting angular velocity for pendulum 2

#----------------------------------------------------------------------------------------------------------------------------------

u_vector = [theta_1_0, theta_2_0, thetaDot_1_0, thetaDot_2_0]

deltaT = float(1 / fps) / float(stepsPerFrame)
u_vectorTimeSnapshots = []
def symplecticEulerOneStep():
    def findThetaDoubleDot_1(theta_1, theta_2, thetaDot_1, thetaDot_2):
        return (-m_2*l_1*thetaDot_1**2*np.sin(theta_1 - theta_2)*np.cos(theta_1 - theta_2) + m_2*g*np.sin(theta_2)*np.cos(theta_1 - theta_2) - m_2*l_2*thetaDot_2**2*np.sin(theta_1 - theta_2) - (m_1 + m_2)*g*np.sin(theta_1)) / ((m_1 + m_2)*l_1 - m_2*l_1*np.cos(theta_1 - theta_2)**2) - frictionCoefficient * np.sign(thetaDot_1)

    def findThetaDoubleDot_2(theta_1, theta_2, thetaDot_1, thetaDot_2):
        return (m_2*l_2*thetaDot_2**2*np.sin(theta_1 - theta_2)*np.cos(theta_1 - theta_2) + (m_1 + m_2)*g*np.sin(theta_1)*np.cos(theta_1 - theta_2) + l_1*thetaDot_1**2*np.sin(theta_1-theta_2)*(m_1 + m_2) - g*np.sin(theta_2)*(m_1 + m_2)) / (l_2*(m_1 + m_2) - m_2*l_2*np.cos(theta_1 - theta_2)**2) - frictionCoefficient * np.sign(thetaDot_2)

    u_vector[2] += findThetaDoubleDot_1(u_vector[0], u_vector[1], u_vector[2], u_vector[3]) * deltaT
    u_vector[3] += findThetaDoubleDot_2(u_vector[0], u_vector[1], u_vector[2], u_vector[3]) * deltaT

    u_vector[0] += u_vector[2] * deltaT
    u_vector[1] += u_vector[3] * deltaT

    u_vectorTimeSnapshots.append(copy.deepcopy(u_vector))

radiusOfGraphAxises = max(l_1, l_2) * 2.1
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
                line.append(l_1*np.sin(u_vectorTimeSnapshots[i][0]) + l_2*np.sin(u_vectorTimeSnapshots[i][1]))
            return line
        if _axis == 1 and _particle == 1:
            for i in range(len(u_vectorTimeSnapshots)):
                line.append(-l_1*np.cos(u_vectorTimeSnapshots[i][0]) - l_2*np.cos(u_vectorTimeSnapshots[i][1]))
            return line

fig = plt.figure()
ax = plt.axes()
plt.gca().set_aspect('equal', adjustable='box')
ax.set_xlim(-radiusOfGraphAxises, radiusOfGraphAxises)
ax.set_ylim(-radiusOfGraphAxises, radiusOfGraphAxises)
plt.rcParams['axes.titley'] = 1
ax.set_facecolor(backgroundColorRGB)
fig.patch.set_facecolor(backgroundColorRGB)
#plt.axis('off')

line0, = ax.plot(0, 0, lw=0.5, color=((0, 0, 1)))
line1, = ax.plot(0, 0, lw=0.5, color=((1, 0, 0)))
bar0, = ax.plot(0, 0, lw=3, color=(lineColor))
bar1, = ax.plot(0, 0, lw=3, color=(lineColor))

counter = 0
scats = []
def animation_frame(i):
    global counter
    counter += 1
    def on_click(event):
        global counter
        counter = 0
        # get the x and y pixel coords
        x, y = event.x, event.y
        if event.inaxes:
            if event.xdata > 0:
                global u_vectorTimeSnapshots
                u_vectorTimeSnapshots = []
                u_vector[2] = 0
                u_vector[3] = 0
                u_vector[0] = np.arctan(event.ydata / event.xdata) + np.pi / 2
                u_vector[1] = np.arctan(event.ydata / event.xdata) + np.pi / 2
            else:
                u_vectorTimeSnapshots = []
                u_vector[2] = 0
                u_vector[3] = 0
                u_vector[0] = np.arctan(event.ydata / event.xdata) + np.pi / 2 + np.pi
                u_vector[1] = np.arctan(event.ydata / event.xdata) + np.pi / 2 + np.pi
    plt.connect('button_press_event', on_click)

    global starttime
    if i == 0:
        starttime = datetime.now()

    for x in range(stepsPerFrame):
        symplecticEulerOneStep()

    averageFps = round(i / ((datetime.now() - starttime).total_seconds()))
    t = (f'Average FPS: {averageFps} || LineTime: {lineTime} s\nBlue: {l_1} m, {m_1} kg || Red: {l_2} m, {m_2} kg || Friction: {frictionCoefficient}')
    ax.set_title(t)

    if counter > fps * lineTime:
        for i in range(stepsPerFrame):
            del u_vectorTimeSnapshots[0]

    line0.set_data(getAxisCoordinatesOverTimeForParticle(0, 0), getAxisCoordinatesOverTimeForParticle(0, 1))

    line1.set_data(getAxisCoordinatesOverTimeForParticle(1, 0), getAxisCoordinatesOverTimeForParticle(1, 1))

    bar0.set_data([0, l_1*np.sin(u_vector[0])], [0, -l_1*np.cos(u_vector[0])])

    bar1.set_data([l_1*np.sin(u_vector[0]), l_1*np.sin(u_vector[0]) + l_2*np.sin(u_vector[1])], [-l_1*np.cos(u_vector[0]), -l_1*np.cos(u_vector[0]) - l_2*np.cos(u_vector[1])])

    global scats
    # first remove all old scatters
    for scat in scats:
        scat.remove()
    scats = []

    scats.append((ax.scatter(l_1*np.sin(u_vector[0]), -l_1*np.cos(u_vector[0]), color=(0, 0, 1), s=50, zorder=2)))
    scats.append((ax.scatter(l_1*np.sin(u_vector[0]) + l_2*np.sin(u_vector[1]), -l_1*np.cos(u_vector[0]) - l_2*np.cos(u_vector[1]), color=(1, 0, 0), s=50, zorder=2)))

ani = animation.FuncAnimation(fig, func=animation_frame, frames=np.arange(0, 100, 1), interval=1000.0 / fps)
plt.show()

#ani.save('doublePendulum3.gif')