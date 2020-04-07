#!/usr/bin/env python
import rospy

from laser.msg import st
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
PI=3.141592

msg = st()

class laser:
	def __init__(self):
		self.mode = None
		self.a = 0
		self.vel_msg = Twist()
		self.avg = 0
		self.sum_r = 0
		self.sum_l = 0
		self.count_r = 0
		self.count_l = 0
		self.velocity_publisher = rospy.Publisher("cmd_vel", Twist, queue_size=10)
	def gos(self):
		self.vel_msg.linear.x = 0.2
		self.vel_msg.angular.z = 0
		self.velocity_publisher.publish(self.vel_msg)

	def turn(self, a):
		self.vel_msg.linear.x = 0
		self.vel_msg.angular.z = a*(np.pi)/6
		self.velocity_publisher.publish(self.vel_msg)

	def stop(self):
		self.vel_msg.linear.x = 0
		self.vel_msg.angular.z = 0
		self.velocity_publisher.publish(self.vel_msg)
	
	#def move(self):
		
	def callback(self, data):
		
		for num_r in data.ranges[1:40]:
			if num_r > 0.02 and num_r < 0.6:		#checking laser value
				self.sum_r = self.sum_r + num_r
				self.count_r = self.count_r + 1
			#else:
			#	self.sum_r = 0
			#	self.count_r = 0
		for num_l in data.ranges[310:340]:
			if num_l >0.02 and num_l < 0.6:
				self.sum_l = self.sum_l + num_l
				self.count_l = self.count_l + 1
			#else:
			#	self.sum_l = 0
			#	self.count_l = 0
		if self.count_r > self.count_l:
			self.avg = self.sum_r / self.count_r
			if self.avg < 0.4:				#average laser value
				self.turn(-1)
				self.mode = "turn_r"
				
				print(self.mode)
			else:
				self.turn(0)
				self.gos()
				self.mode = "gos"
				print(self.mode)
		elif self.count_l > self.count_r:
			self.avg = self.sum_l / self.count_l
			if self.avg < 0.4:				#average laser value
				self.turn(1)
				self.mode = "turn_l"
				
				print(self.mode)
			else:
				self.turn(0)
				self.gos()
				self.mode = "gos"
				print(self.mode)
		else:
			self.turn(0)
			self.gos()
			self.mode = "gos"
			print(self.mode)
		self.sum_r = 0
		self.count_r = 0
		self.sum_l = 0
		self.count_l = 0
			
		
		
			#velocity_publisher.publish(vel.msg)
		
		
def main():
	move = laser()
	rospy.init_node("laser_move")
	
	rospy.Subscriber("/scan", LaserScan, move.callback)
	vel_msg = Twist()
	rate = rospy.Rate(10)
	
	while not rospy.is_shutdown():
		move.a += 1
			
	rate.sleep()

if __name__ == '__main__':
	try :
		main()
	except rospy.ROSInterruptException:
		pass

