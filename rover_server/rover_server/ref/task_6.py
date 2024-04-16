'''
*****************************************************************************************
*
*        		===============================================
*           		Pharma Bot (PB) Theme (eYRC 2022-23)
*        		===============================================
*
*  This script is to implement Task 6 of Pharma Bot (PB) Theme (eYRC 2022-23).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			[ 2139 ]
# Author List:		[ Yazid Musthafa]
# Filename:			task_6.py
# Functions:		
# 					[ distance, search_pickedup, search_visited, deepcopy, deepcopy_2, path_planning, paths_to_moves, perspective_transform, transform_values, over_head_cam, o_h_c, emulation, set_values, sort_dest, sort_destination]
# Filename: Task_6.py
# Theme: PharmaBot
# Global Variables: [N, E, S, W, head, enter, x, y, h, w, X_axis, delivery_loc, pts1, pts2, c_x, c_y, angle, scene_parameters, graph, shops, emulate]

####################### IMPORT MODULES #######################
## You are not allowed to make any changes in this section. ##
## You have to implement this task with the three available ##
## modules for this task (numpy, opencv)                    ##
##############################################################
import socket
import time
import os, sys
from zmqRemoteApi import RemoteAPIClient
import traceback
import zmq
import numpy as np
import cv2
from pyzbar.pyzbar import decode
import json
import random
import task_1b
import ast
import threading
##############################################################

N = 1
E = 2
S = 3
W = 4
head = N
enter = True
x = 410
y = 60
h = 1190
w = 1075

X_axis = ['A','B','C','D','E','F']

delivery_loc = [0,[0.12,-0.07],[-0.07,-0.12],[-0.12, 0.07],[0.07, 0.12]]

## Import PB_theme_functions code
try:
	pb_theme = __import__('PB_theme_functions')

except ImportError:
	print('\n[ERROR] PB_theme_functions.py file is not present in the current directory.')
	print('Your current directory is: ', os.getcwd())
	print('Make sure PB_theme_functions.py is present in this current directory.\n')
	sys.exit()
	
except Exception as e:
	print('Your PB_theme_functions.py throwed an Exception, kindly debug your code!\n')
	traceback.print_exc(file=sys.stdout)
	sys.exit()
x_coordinates = [0,'A','B','C','D','E','F','G']
visited = []
pickedup = []
emulate = True

def distance(x,y):
	x1 = x_coordinates.index(x[0])
	y1 = x[1]
	x2 = x_coordinates.index(y[0])
	y2 = y[1]

	#print(x1,'   ',x2,'   ',y1,'   ',y2)

	dist = abs(int(x2) - int(x1)) + abs(int(y2) - int(y1))

	return dist

def search_pickedup(pack):
	for p in pickedup:
		if p == pack:
			x = 1
			break
		else:
			x = 0
	return x

def search_visited(node):
	for nd in visited:
		if nd == node:
			x = 1
			break
		else:
			x = 0
	return x

def deepcopy(old_list):
	new_list = [ i[:] for i in old_list ]
	return new_list

def deepcopy_2(old_list):
	new_list = [ i for i in old_list ]
	return new_list

def map_range(x, in_min, in_max, out_min, out_max):
	return (((x - in_min) * (out_max - out_min)) / (in_max - in_min)) + out_min


def path_planning(graph, start, end):

	"""
	Purpose:
	---
	This function takes the graph(dict), start and end node for planning the shortest path

	** Note: You can use any path planning algorithm for this but need to produce the path in the form of 
	list given below **

	Input Arguments:
	---
	`graph` :	{ dictionary }
			dict of all connecting path
	`start` :	str
			name of start node
	`end` :		str
			name of end node


	Returns:
	---
	`backtrace_path` : [ list of nodes ]
			list of nodes, produced using path planning algorithm

		eg.: ['C6', 'C5', 'B5', 'B4', 'B3']
	
	Example call:
	---
	arena_parameters = detect_arena_parameters(maze_image)
	"""    

	backtrace_path=[]

	##############	ADD YOUR CODE HERE	##############

	prior_1 = []
	prior_2 = []
	check_set = []
	paths = []

	dest = False
	curr_check = list(graph[start].keys())                              ## Assigning primary priority path set  prior_1 and prior_2
	for i in range (len(curr_check)):
		if len(curr_check) > 1:
			check_set.append(curr_check[i])
			dist = distance(check_set[i],end)
			check_set[i] = [dist,check_set[i]]	
			if i == len(curr_check)-1:
				check_set.sort()
				if check_set[0][0] == check_set[1][0]:
					for k in range (2):
						prior_1.append([check_set[k][1]])
					for j in range (2,len(check_set)):
						prior_2.append([check_set[j][1]])
				else:
					prior_1.append([check_set[0][1]])
					for j in range (1,len(check_set)):
						prior_2.append([check_set[j][1]])

		if len(curr_check) == 1:
			prior_1.append([curr_check[0]])

		if curr_check[i] == end:
			backtrace_path = [[start,curr_check[i]]]
			dest = True

	
	new_path = []
	check_set.clear()
	visited.clear()
	visited.append(start)

	while dest == False:                                            ####################
		p1_temp = deepcopy(prior_1)
		for h in range (len(prior_1)):
			curr_check = list(graph[prior_1[h][-1]].keys())
			if len(prior_1[h]) > 1:
				curr_check.remove(prior_1[h][-2])
			elif len(prior_1[h]) == 1:
				curr_check.remove(start)
			if len(curr_check) == 0:
				p1_temp.remove(prior_1[h])
			elif len(curr_check) != 0:
				check_set = []
				dist1 = distance(prior_1[h][-1],end)
				for nod in curr_check:
					dist2 = distance(nod,end)
					node_check = search_visited(nod)
					if (dist1 < dist2 and node_check == 1):
						curr_check.remove(nod)
				for i in range (len(curr_check)):
					dist2 = distance(curr_check[i],end)
					node_check = search_visited(curr_check[i])
					if (dist1 > dist2 and node_check == 1) or node_check == 0:
						if len(curr_check) > 1:
							check_set.append(curr_check[i])
							dist = distance(check_set[-1],end)
							check_set[-1] = [dist,check_set[-1]]
							if i == len(curr_check)-1:
								check_set.sort()
								if len(check_set)>1:
									if check_set[0][0] == check_set[1][0]:
										new_path = deepcopy_2(prior_1[h])
										p1_temp[p1_temp.index(prior_1[h])].append(check_set[0][1])
										new_path_2 = deepcopy_2(new_path)
										p1_temp.append(new_path)

										p1_temp[p1_temp.index(new_path)].append(check_set[1][1])
										for j in range (2,len(check_set)):
											new_path_3 = deepcopy_2(new_path_2)
											prior_2.append(new_path_3)
											prior_2[prior_2.index(new_path_3)].append(check_set[j][1])
											if len(prior_1[h]) == 1 and prior_2[-1][0] != start:
												prior_2[-1].insert(0,start)
																								
									else:
										new_path_2 = deepcopy_2(prior_1[h])
										p1_temp[p1_temp.index(prior_1[h])].append(check_set[0][1])
										for j in range (1,len(check_set)):
											new_path_3 = deepcopy_2(new_path_2)
											prior_2.append(new_path_3)
											prior_2[prior_2.index(new_path_3)].append(check_set[j][1])
											if len(prior_1[h]) == 1 and prior_2[-1][0] != start:
												prior_2[-1].insert(0,start)
												
								if len(check_set) == 1:
									p1_temp[p1_temp.index(prior_1[h])].append(check_set[0][1])
							if dist == 0:
								for z in range(len(p1_temp)):
									if p1_temp[z][-1] != end and prior_1[h] == p1_temp[z]:
										p1_temp[z].append(curr_check[i])	
						if len(curr_check) == 1:
							p1_temp[p1_temp.index(prior_1[h])].append(curr_check[0])
					if curr_check[i] == end:
						break
			visited.append(prior_1[h][-1])
			for z in range(len(p1_temp)):
				if len(prior_1[h]) == 1 and p1_temp[z][0] != start and p1_temp[z][0] == prior_1[h][0]:
					p1_temp[z].insert(0,start)
			if h == len(prior_1)-1:
				for z in range(len(p1_temp)):
					if p1_temp[z][-1] == end:
						paths.append(p1_temp[z])
						p1_temp[z] = []
				pr = deepcopy(p1_temp)
				for p in range(len(p1_temp)):
					if p1_temp[p] == []:
						pr.remove([])
				p1_temp = deepcopy(pr)

		if prior_1 == p1_temp:
			p1_temp = []
		prior_1 = deepcopy(p1_temp)

		if prior_1 == []:
			prior_1 = deepcopy(prior_2)
			prior_2.clear()

		if (len(prior_1)==0 and len(prior_2) == 0) or len(paths) > 2:
			dest = True
			break

	paths.sort(key = len)
	if len(paths) > 1:
		for i in range(len(paths)):
			if len(paths[i]) == len(paths[0]):
				backtrace_path.append(paths[i])
			else:
				break
	if len(paths) == 1:
		backtrace_path = deepcopy(paths)
	
	##################################################

	return backtrace_path

def paths_to_moves(paths, traffic_signal):
	global head

	"""
	Purpose:
	---
	This function takes the list of all nodes produces from the path planning algorithm
	and connecting both start and end nodes

	Input Arguments:
	---
	`paths` :	[ list of all nodes ]
			list of all nodes connecting both start and end nodes (SHORTEST PATH)
	`traffic_signal` : [ list of all traffic signals ]
			list of all traffic signals
	---
	`moves` : [ list of moves from start to end nodes ]
			list containing moves for the bot to move from start to end

			Eg. : ['UP', 'LEFT', 'UP', 'UP', 'RIGHT', 'DOWN']
	
	Example call:
	---
	moves = paths_to_moves(paths, traffic_signal)
	"""  

	list_moves=[]

	##############	ADD YOUR CODE HERE	##############
	
	straight = 1
	left = 0
	right = 2
	wait_5 = 3

	dir = ['LEFT','STRAIGHT','RIGHT','WAIT_5']
	count = -1
	TS_cnt = []
	move = []
	if len(paths) > 1:
		for pth in paths:
			count += 1
			TS_cnt.append(0)
			for i in range(len(pth)):
				for j in range(len(traffic_signal)):
					if pth[i] == traffic_signal[j]:
						TS_cnt[count] += 1
		cnt = deepcopy_2(TS_cnt)
		cnt.sort()
		path_temp = deepcopy_2(paths[TS_cnt.index(cnt[0])])
		path_final = deepcopy_2(path_temp)

	if len(paths) == 1:
		path_temp = deepcopy_2(paths[0])
		path_final = deepcopy_2(path_temp)

	for path in path_temp:
		path_temp[path_temp.index(path)] = int(x_coordinates.index(path[0]))+(6*(int(path[1])-1))	
	
	if (path_temp[0] - path_temp[1]) == -6 and head == N:
		if path_temp[0] % 6 != 1:
			move = [0,'LEFT','LEFT']
			head = S
		if path_temp[0] % 6 == 1:
			move = [0,'RIGHT','RIGHT']
			head = S

	if (path_temp[0] - path_temp[1]) == 6 and head == S:
		if path_temp[0] % 6 != 0:
			move = [0,'LEFT','LEFT']
			head = N
		if path_temp[0] % 6 == 0:
			move = [0,'RIGHT','RIGHT']
			head = N

	if (path_temp[0] - path_temp[1]) == 1 and head == E:
		if path_temp[0] > 6:
			move = [0,'LEFT','LEFT']
			head = W
		if path_temp[0] <= 6:
			move = [0,'RIGHT','RIGHT']
			head = W

	if (path_temp[0] - path_temp[1]) == -1 and head == W:
		if path_temp[0] > 6:
			move = [0,'RIGHT','RIGHT']
			head = E
		if path_temp[0] <= 6:
			move = [0,'LEFT','LEFT']
			head = E

	if move != []:
		move = str(move)
		pb_theme.send_message_via_socket(connection_2, move)
		msg = None
		while msg != "DONE":
			msg = pb_theme.receive_message_via_socket(connection_2)
		move = []

	for j in range(len(path_temp)):
		
		pos = path_temp[j]
		if j != len(path_temp)-1:
			next = path_temp[j+1]

			if (abs(pos - next) == 6 and (head == N or head == S)) or (abs(pos - next) == 1 and (head == E or head == W)):
				turn = straight

			if ((pos - next) == 6 and head == W) or (head == N and (pos - next) == -1) or (head == E and (pos - next) == -6) or (head == S and (pos - next) == 1):
				turn = right

			if ((pos - next) == 6 and head == E) or (head == N and (pos - next) == 1) or (head == W and (pos - next) == -6) or (head == S and (pos - next) == -1):
				turn = left

			if turn == left:
				head = head - 1
			if turn == right:
				head = head + 1
			if head == 0:
				head = W
			if head == 5:
				head = N

			for pnt in traffic_signal:

				point = int(x_coordinates.index(pnt[0]))+(6*(int(pnt[1])-1))
				if point == pos:
					list_moves.append(dir[wait_5])

			list_moves.append(dir[turn])

	##################################################

	return list_moves, path_final
#####################################################################################

def perspective_transform(image):

	global enter, pts1, pts2

	"""
    Purpose:
    ---
    This function takes the image as an argument and returns the image after 
    applying perspective transform on it. Using this function, you should
    crop out the arena from the full frame you are receiving from the 
    overhead camera feed.

    HINT:
    Use the ArUco markers placed on four corner points of the arena in order
    to crop out the required portion of the image.

    Input Arguments:
    ---
    `image` :	[ numpy array ]
            numpy array of image returned by cv2 library 

    Returns:
    ---
    `warped_image` : [ numpy array ]
            return cropped arena image as a numpy array
    
    Example call:
    ---
    warped_image = perspective_transform(image)
    """  

	warped_image = [] 
	frame = image

	frame = frame[y:y+h, x:x+w]
		
	cameraMatrix = np.array([[817.59874821,   0.,         510.1051283 ],[  0.,         817.69148315, 485.31688792],[  0.,           0.,           1.        ]])
	dist = np.array([[-0.21835082,  0.03802973, -0.00430877,  0.01848059, -0.00543478]])
	newCameraMatrix = np.array([[597.37365723,   0.,         578.46570874],[  0.,         571.41943359, 461.37368269],[  0.,           0.,           1.        ]])

	# Undistort
	dst = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)

	if enter == True:
		a,b = task_1b.detect_ArUco_details(dst)
		z = list(a.values())
		#print(z)

		if len(z) >= 4:
			pts1 = np.float32([[a[3][0][0], a[3][0][1]], [a[4][0][0], a[4][0][1]], [a[2][0][0], a[2][0][1]], [a[1][0][0], a[1][0][1]]])
			pts2 = np.float32([[0, 0], [1025, 0], [0, 984], [1025, 984]])
			enter = False

	# Apply Perspective Transform Algorithm
	matrix = cv2.getPerspectiveTransform(pts1, pts2)
	warped_image = cv2.warpPerspective(dst, matrix, (1025, 984))
	
	#cv2.imshow('frame', dst)
	#cv2.waitKey(1)

	return warped_image

#####################################################################################

def transform_values(image):
	global c_x, c_y, angle

	"""
    Purpose:
    ---
    This function takes the image as an argument and returns the 
    position and orientation of the ArUco marker (with id 5), in the 
    CoppeliaSim scene.

    Input Arguments:
    ---
    `image` :	[ numpy array ]
            numpy array of image returned by camera

    Returns:
    ---
    `scene_parameters` : [ list ]
            a list containing the position and orientation of ArUco 5
            scene_parameters = [c_x, c_y, c_angle] where
            c_x is the transformed x co-ordinate [float]
            c_y is the transformed y co-ordinate [float]
            c_angle is the transformed angle [angle]
    
    HINT:
        Initially the image should be cropped using perspective transform 
        and then values of ArUco (5) should be transformed to CoppeliaSim
        scale.
    
    Example call:
    ---
    scene_parameters = transform_values(image)
    """   
 
	scene_parameters = []
 
	a,_ = task_1b.detect_ArUco_details(image)
	if 5 in a.keys():
		x = a[5][0][0]
		c_x = map_range(x,0,1025, -0.9550, 0.9550)
		y = a[5][0][1]
		c_y = map_range(y,0,984, 0.9550, -0.9550)
		angle = a[5][1]
	scene_parameters.append(c_x)
	scene_parameters.append(c_y)
	pi=22/7
	radian = angle*(pi/180)
	scene_parameters.append(radian)

	return scene_parameters

#####################################################################################

def over_head_cam():
	t = threading.Thread(target=o_h_c)
	t.start()

def emulation():
	t = threading.Thread(target=set_values)
	t.start()
	
def o_h_c():
	global scene_parameters

	scene_parameters = []

	cap = cv2.VideoCapture(1)
	frameWidth = 1920
	frameHeight = 1080
	cap.set(3, frameWidth)
	cap.set(4, frameHeight)
	cap.set(10, 150)

	while True:

		ret, frame = cap.read()

		result = perspective_transform(frame)

		scene_param = transform_values(result)

		if scene_param != []:
			scene_parameters = scene_param

		cv2.imshow('frame', result)
		cv2.waitKey(1)

def set_values():
	"""
    Purpose:
    ---
    This function takes the scene_parameters, i.e. the transformed values for
    position and orientation of the ArUco marker, and sets the position and 
    orientation in the CoppeliaSim scene.

    Input Arguments:
    ---
    `scene_parameters` :	[ list ]
            list of co-ordinates and orientation obtained from transform_values()
            function

    Returns:
    ---
    None

    HINT:
        Refer Regular API References of CoppeliaSim to find out functions that can
        set the position and orientation of an object.
    
    Example call:
    ---
    set_values(scene_parameters)
    """   
	global emulate, scene_parameters
	while True:
		if emulate == True:
			sim.setObjectOrientation(aruco_handle, -1, [-1.5812642574310303, -scene_parameters[2], -1.5707961320877075])
			posi = [scene_parameters[0],scene_parameters[1], 0.048]
			sim.setObjectPosition(aruco_handle, arena_handle, posi)

#####################################################################################
def dest_path(paths, traffic_signal):
	count = -1
	TS_cnt = []
	if len(paths) > 1:
		for pth in paths:
			count += 1
			TS_cnt.append(0)
			for i in range(len(pth)):
				for j in range(len(traffic_signal)):
					if pth[i] == traffic_signal[j]:
						TS_cnt[count] += 1
		cnt = deepcopy_2(TS_cnt)
		cnt.sort()
		path_temp = paths[TS_cnt.index(cnt[0])]

	if len(paths) == 1:
		path_temp = deepcopy_2(paths[0])
	
	return path_temp
	
#################################################################################

def sort_dest(start, qr_message):
	"""
	Purpose:
	---
	This function sorts the destinations of packages

	Input Arguments:
	---
    `start` : starting node

	'qr_message' : [ object ]

	Returns:
	---
	final
	
	Example call:
	---
	sort_des(start, qr_message, needed)
	"""
	final_dests = []

	if len(qr_message) <= 2 or len(shops) != 0:
		while True:
			for i in range(len(qr_message)):
				paths = path_planning(graph, start, qr_message[i][1])
				paths = dest_path(paths, traffic_signals)
				qr_message[i] = (len(paths),) + qr_message[i]
			
			qr_message = sorted(qr_message)
			final_dests.append(qr_message[0])
			qr_message.remove(qr_message[0])

			for i in range(len(qr_message)):
				qr_message[i] = (qr_message[i][1],qr_message[i][2])

			if len(qr_message) == 0:
				qr_message = deepcopy_2(final_dests)
				break
			start = final_dests[-1][2]

	elif len(qr_message) == 3 and len(shops) == 0:
		for i in range(len(qr_message)):
			paths = path_planning(graph, qr_message[i][1], end_node)
			paths = dest_path(paths, traffic_signals)
			qr_message[i] = (len(paths),) + qr_message[i]

		qr_message = sorted(qr_message)
		final_dests.append(qr_message[-1])
		qr_message.remove(qr_message[-1])

		for i in range(len(qr_message)):
			qr_message[i] = (qr_message[i][1],qr_message[i][2])

		for i in range(len(qr_message)):
			paths = path_planning(graph, start, qr_message[i][1])
			paths = dest_path(paths, traffic_signals)
			qr_message[i] = (len(paths),) + qr_message[i]
		
		qr_message = sorted(qr_message)
		final_dests.insert(0,qr_message[1])
		final_dests.insert(0,qr_message[0])

		paths = path_planning(graph, start, final_dests[0][2])
		paths = dest_path(paths, traffic_signals)

		dest_corrected = False

		for nd in paths:
			if nd == final_dests[-1][2]:
				dest_corrected = True
				final_dests.insert(0,final_dests.pop(-1))

				paths = path_planning(graph, final_dests[-1][2], end_node)
				paths = dest_path(paths, traffic_signals)

				for nd in paths:
					if nd == final_dests[1][2]:
						final_dests[-1],final_dests[1] = final_dests[1],final_dests[-1]
						break
				break
		
		if dest_corrected == False:
			paths = path_planning(graph, final_dests[1][2], final_dests[2][2])
			paths = dest_path(paths, traffic_signals)

			for nd in paths:
				if nd == final_dests[0][2]:
					final_dests[0],final_dests[1] = final_dests[1],final_dests[0]
					break

		qr_message = deepcopy(final_dests)

	if len(qr_message) == 2 and len(shops) == 0:
		paths = path_planning(graph, final_dests[1][2], end_node)
		paths = dest_path(paths, traffic_signals)

		for nd in paths:
			if nd == final_dests[0][2]:
				final_dests[0],final_dests[1] = final_dests[1],final_dests[0]
				break

	for i in range(len(final_dests)):
		final_dests[i] = (final_dests[i][1],final_dests[i][2])
	
	return final_dests

#####################################################################################

def sort_destination(start, qr_message, needed):

	"""
	Purpose:
	---
	This function sorts the destinations of packages

	Input Arguments:
	---
    `start` : starting node

	    
	'qr_message' : [ object ]

	'needed' : [ int ]

	Returns:
	---
	final
	
	Example call:
	---
	sort_destination(start, qr_message, needed)
	"""
	
	final_dests = []
	final = []
	qr_msg = deepcopy(qr_message)
	while True:
		for i in range(len(qr_message)):
			paths = path_planning(graph, start, qr_message[i][1])
			paths = dest_path(paths, traffic_signals)
			qr_message[i] = (len(paths),) + qr_message[i]
		
		qr_message = sorted(qr_message)
		final_dests.append(qr_message[0])
		qr_message.remove(qr_message[0])

		for i in range(len(qr_message)):
			qr_message[i] = (qr_message[i][1],qr_message[i][2])

		if len(qr_message) == 0:
			break
		start = final_dests[-1][2]
	
	for i in range(len(final_dests)):
		final_dests[i] = (final_dests[i][1],final_dests[i][2])
		if i <= needed-1:
			pickedup.append(final_dests[i])
			final.append(final_dests[i])

	if needed >= len(qr_msg):
		shops.remove(shops[0])
	
	return final

#####################################################################################

def task_6_implementation(sim, maze_image):
	global scene_parameters, graph, shops, emulate
	"""
	Purpose:
	---
	This function contains the implementation logic for task 6 

	Input Arguments:
	---
    `sim` : [ object ]
            ZeroMQ RemoteAPI object
	    
	'maze_image' : [ image ]

	Returns:
	---
	None
	
	Example call:
	---
	task_5_implementation(sim)
	"""

	##################	ADD YOUR CODE HERE	##################
	
	arena_handle = sim.getObject('/Arena')
	aruco_handle = sim.getObject('/AlphaBot')

	count = 0
	graph = pb_theme.detect_paths_to_graph(maze_image)
	shop_pickup = {'Shop_1':'B2','Shop_2':'C2','Shop_3':'D2','Shop_4':'E2','Shop_5':'F2'}
	shops = []

	for s in range(len(medicine_package_details)):
		shop = medicine_package_details[s][0]
		if s == 0:
			pickup = shop_pickup[shop]
			shops.append(pickup)
		elif shop != medicine_package_details[s-1][0]:
			pickup = shop_pickup[shop]
			shops.append(pickup)

	paths1 = path_planning(graph, start_node, shops[0])
	paths1 = dest_path(paths1, traffic_signals)
	paths2 = path_planning(graph, start_node, shops[-1])
	paths2 = dest_path(paths2, traffic_signals)

	if len(paths1) > len(paths2):
		shops = sorted(shops, reverse=True)

	end = shops[0]
	temp_qr_message = []

	needed = 3
	fst = True
	
	loc = [0.0,0.03,0.06]
	d = 0
	qr_message = None
	
	paths = path_planning(graph, start_node, end)
	moves, path = paths_to_moves(paths, traffic_signals)
	moves = str(moves)

	### 
	if scene_parameters != []:
		emulate = True
		emulation()
	
	while scene_parameters == []:
		if scene_parameters != []:
			emulate = True
			emulation()

	pb_theme.send_message_via_socket(connection_2, moves)
	while True:
		
		message = pb_theme.receive_message_via_socket(connection_2)
		#print(message)
		
		'''if message == "node":
			count += 1
			node_arrive = "ARRIVED AT: "+str(path[count])
			print(node_arrive)
			time.sleep(0.5)
			
			pb_theme.send_message_via_socket(connection_1, node_arrive)
		if message == "Wait":
			node_arrive = "WAIT AT: "+path[count]
			print(node_arrive)
			pb_theme.send_message_via_socket(connection_1, node_arrive)'''
		if message == "destination":
			if path[-1] == end_node:
				emulate = False
				time.sleep(0.5)
				break
			if path[-1] == end and fst == True:
				d = 0

				qr_plane = '/qr_plane_'+ str(x_coordinates.index(end[0])-1)
				emulate = False
				time.sleep(1)

				QrHandle = sim.getObject(qr_plane)

				sim.setObjectInt32Parameter(QrHandle, sim.objintparam_visibility_layer, 65536)

				emulate = True
				time.sleep(1.5)
				emulate = False
				time.sleep(0.7)

				while qr_message == None:
					qr_message = pb_theme.read_qr_code(sim)


				if qr_message != None:
					qr_message = ast.literal_eval(qr_message)
					qr_message = list(qr_message.items()) 
				
				sim.setObjectInt32Parameter(QrHandle, sim.objintparam_visibility_layer, 0)

				emulate = True

				#qr_message = sorted(qr_message, key=lambda x: x[1])                      ######################################
				if pickedup != []:
					temp_qr = []
					for p in qr_message:
						x = search_pickedup(p)
						if x == 0:
							temp_qr.append(p)
					
					qr_message = deepcopy(temp_qr)

				qr_message = sort_destination(end, qr_message, needed)
				node_arrive = "ARRIVED AT: "+end

				emulate = False
				time.sleep(0.5)
				
				for i in range(len(qr_message)):

					objectHandle = sim.getObjectHandle(qr_message[i][0])
					if len(qr_message) < 3:
						sim.setObjectPosition(objectHandle, aruco_handle , [0.03,0.0,loc[len(temp_qr_message)+i]])
					if len(qr_message) == 3:
						sim.setObjectPosition(objectHandle, aruco_handle , [0.03,0.0,loc[i]])
					
					sim.setObjectParent(objectHandle, aruco_handle , True)
					
					send_msg = str([qr_message[i][0]])
					pb_theme.send_message_via_socket(connection_2, send_msg)

					pack = qr_message[i][0].split("_")
					pack_details = "PICKED UP: "+pack[0]+", "+pack[1]+", "+qr_message[i][1]
					pb_theme.send_message_via_socket(connection_1, pack_details)
					print(pack_details)
				
				emulate = True

				for i in range(len(qr_message)):
					qr_temp = deepcopy(qr_message)
					temp_qr_message.append(qr_temp[i])

				if len(temp_qr_message) < 3 and len(shops) != 0:
					needed = 3 - len(temp_qr_message)
					go = False
					pb_theme.send_message_via_socket(connection_1, node_arrive)
					paths = path_planning(graph, end, shops[0])
					moves, path = paths_to_moves(paths, traffic_signals)
					moves = str(moves)
					time.sleep(0.2)
					pb_theme.send_message_via_socket(connection_2, moves)
					end = shops[0]
					qr_message = None
					count = 0

				if len(temp_qr_message) == 3 or (len(temp_qr_message) < 3 and len(shops) == 0):
					qr_message = sort_dest(end, temp_qr_message)
					go = True

				if go == True:		
					pb_theme.send_message_via_socket(connection_1, node_arrive)
					paths = path_planning(graph, end, qr_message[0][1])
					moves, path = paths_to_moves(paths, traffic_signals)
					moves = str(moves)
					time.sleep(0.2)
					pb_theme.send_message_via_socket(connection_2, moves)
					count = 0
					fst = False

			elif qr_message:
				if path[-1] == qr_message[d][1]:
					#if qr_message[d][1] != end_node:
					ledOff = []
					for ld in range(d+1):
						if path[-1] == qr_message[ld][1]:								
							pos_x = 0.89-(X_axis.index(qr_message[ld][1][0])*0.356)
							pos_Y = -0.89+((int(qr_message[ld][1][1])-1)*0.356)

							emulate = False
							time.sleep(0.5)
							
							objectHandle = sim.getObjectHandle(qr_message[ld][0])
							sim.setObjectParent(objectHandle, arena_handle , True)
							sim.setObjectPosition(objectHandle, -1 , [pos_x + delivery_loc[head][0],pos_Y + delivery_loc[head][1],0.016])

							emulate = True

							ledOff.append(qr_message[ld][0])
							pack = qr_message[ld][0].split("_")
							pack_details = "DELIVERED: "+pack[0]+", "+pack[1]+" AT "+qr_message[ld][1]
							pb_theme.send_message_via_socket(connection_1, pack_details)
							print(pack_details)
						else:
							ledOff.append("none")
					ledOff = str(ledOff)
					pb_theme.send_message_via_socket(connection_2, ledOff)
					d += 1
					#node_arrive = "ARRIVED AT: "+qr_message[d-1][1]
					#pb_theme.send_message_via_socket(connection_1, node_arrive)
					
					if d < len(qr_message):
						paths = path_planning(graph, qr_message[d-1][1], qr_message[d][1])
						moves, path = paths_to_moves(paths, traffic_signals)
						moves = str(moves)
						time.sleep(0.2)
						pb_theme.send_message_via_socket(connection_2, moves)
					if d == len(qr_message):
						if len(shops) == 0:
							paths = path_planning(graph, qr_message[-1][1], end_node)
							moves, path = paths_to_moves(paths, traffic_signals)
						else:
							paths1 = path_planning(graph, qr_message[-1][1], shops[0])
							paths1 = dest_path(paths1, traffic_signals)
							paths2 = path_planning(graph, qr_message[-1][1], shops[-1])
							paths2 = dest_path(paths2, traffic_signals)

							if len(paths1) > len(paths2):
								shops = sorted(shops, reverse=True)
							
							end = shops[0]

							paths = path_planning(graph, qr_message[-1][1], end)
							moves, path = paths_to_moves(paths, traffic_signals)

							temp_qr_message = []
							qr_message = None
							count = 0
							needed = 3
							fst = True

						moves = str(moves)
						time.sleep(0.2)
						pb_theme.send_message_via_socket(connection_2, moves)
					count = 0

	##################	ADD YOUR CODE HERE	##################
	

	##########################################################

if __name__ == "__main__":
	
	host = ''
	port = 5050


	## Set up new socket server
	try:
		server = pb_theme.setup_server(host, port)
		print("Socket Server successfully created")

		# print(type(server))

	except socket.error as error:
		print("Error in setting up server")
		print(error)
		sys.exit()


	## Set up new connection with a socket client (PB_task3d_socket.exe)
	try:
		print("\nPlease run PB_socket.exe program to connect to PB_socket client")
		connection_1, address_1 = pb_theme.setup_connection(server)
		print("Connected to: " + address_1[0] + ":" + str(address_1[1]))

	except KeyboardInterrupt:
		sys.exit()


	# ## Set up new connection with a socket client (socket_client_rgb.py)
	# ## Set up new connection with Raspberry Pi
	try:
		print("\nPlease connect to Raspberry pi client")
		connection_2, address_2 = pb_theme.setup_connection(server)
		print("Connected to: " + address_2[0] + ":" + str(address_2[1]))

	except KeyboardInterrupt:
		sys.exit()

	## Send setup message to PB_socket
	pb_theme.send_message_via_socket(connection_1, "SETUP")

	message = pb_theme.receive_message_via_socket(connection_1)
	## Loop infinitely until SETUP_DONE message is received
	while True:
		if message == "SETUP_DONE":
			break
		else:
			print("Cannot proceed further until SETUP command is received")
			message = pb_theme.receive_message_via_socket(connection_1)


	try:
		
		# obtain required arena parameters
		image_filename = os.path.join(os.getcwd(), "config_image.png")
		config_img = cv2.imread(image_filename)
		detected_arena_parameters = pb_theme.detect_arena_parameters(config_img)			
		medicine_package_details = detected_arena_parameters["medicine_packages"]
		traffic_signals = detected_arena_parameters['traffic_signals']
		start_node = detected_arena_parameters['start_node']
		end_node = detected_arena_parameters['end_node']
		horizontal_roads_under_construction = detected_arena_parameters['horizontal_roads_under_construction']
		vertical_roads_under_construction = detected_arena_parameters['vertical_roads_under_construction']

		# print("Medicine Packages: ", medicine_package_details)
		# print("Traffic Signals: ", traffic_signals)
		# print("Start Node: ", start_node)
		# print("End Node: ", end_node)
		# print("Horizontal Roads under Construction: ", horizontal_roads_under_construction)
		# print("Vertical Roads under Construction: ", vertical_roads_under_construction)
		# print("\n\n")

	except Exception as e:
		print('Your task_1a.py throwed an Exception, kindly debug your code!\n')
		traceback.print_exc(file=sys.stdout)
		sys.exit()

	try:

		## Connect to CoppeliaSim arena
		coppelia_client = RemoteAPIClient()
		sim = coppelia_client.getObject('sim')

		## Define all models
		all_models = []

		over_head_cam()
		time.sleep(2)

		## Setting up coppeliasim scene
		print("[1] Setting up the scene in CoppeliaSim")
		all_models = pb_theme.place_packages(medicine_package_details, sim, all_models)
		all_models = pb_theme.place_traffic_signals(traffic_signals, sim, all_models)
		all_models = pb_theme.place_horizontal_barricade(horizontal_roads_under_construction, sim, all_models)
		all_models = pb_theme.place_vertical_barricade(vertical_roads_under_construction, sim, all_models)
		all_models = pb_theme.place_start_end_nodes(start_node, end_node, sim, all_models)
		print("[2] Completed setting up the scene in CoppeliaSim")
		print("[3] Checking arena configuration in CoppeliaSim")

	except Exception as e:
		print('Your task_4a.py throwed an Exception, kindly debug your code!\n')
		traceback.print_exc(file=sys.stdout)
		sys.exit()

	pb_theme.send_message_via_socket(connection_1, "CHECK_ARENA")

	## Check if arena setup is ok or not
	message = pb_theme.receive_message_via_socket(connection_1)
	while True:
		# message = pb_theme.receive_message_via_socket(connection_1)

		if message == "ARENA_SETUP_OK":
			print("[4] Arena was properly setup in CoppeliaSim")
			break
		elif message == "ARENA_SETUP_NOT_OK":
			print("[4] Arena was not properly setup in CoppeliaSim")
			connection_1.close()
			# connection_2.close()
			server.close()
			sys.exit()
		else:
			pass

	## Send Start Simulation Command to PB_Socket
	pb_theme.send_message_via_socket(connection_1, "SIMULATION_START")
	
	## Check if simulation started correctly
	message = pb_theme.receive_message_via_socket(connection_1)
	while True:
		# message = pb_theme.receive_message_via_socket(connection_1)

		if message == "SIMULATION_STARTED_CORRECTLY":
			print("[5] Simulation was started in CoppeliaSim")
			break

		if message == "SIMULATION_NOT_STARTED_CORRECTLY":
			print("[5] Simulation was not started in CoppeliaSim")
			sys.exit()

	pb_theme.send_message_via_socket(connection_2, "START")

	aruco_handle = sim.getObject('/AlphaBot')
	arena_handle = sim.getObject('/Arena')

	task_6_implementation(sim, config_img)


	## Send Stop Simulation Command to PB_Socket
	pb_theme.send_message_via_socket(connection_1, "SIMULATION_STOP")

	## Check if simulation started correctly
	message = pb_theme.receive_message_via_socket(connection_1)
	while True:
		# message = pb_theme.receive_message_via_socket(connection_1)

		if message == "SIMULATION_STOPPED_CORRECTLY":
			print("[6] Simulation was stopped in CoppeliaSim")
			break

		if message == "SIMULATION_NOT_STOPPED_CORRECTLY":
			print("[6] Simulation was not stopped in CoppeliaSim")
			sys.exit()