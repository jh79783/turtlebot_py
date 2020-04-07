#!/usr/bin/env python 
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image , LaserScan
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from newmode.msg import mode_msg, twist, msg_park
bridge = CvBridge()

class Park:
	def __init__(self):
		
		
		self.vel_tw =Twist()
		self.twistpub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
		self.modesub = rospy.Subscriber('/mode_msg', mode_msg, self.modemsg)
		self.lasersub = rospy.Subscriber("/scan", LaserScan, self.callback)
		self.pubpark  =rospy.Publisher('/parkend',msg_park, queue_size = 10)
		self.park_end = msg_park()
		self.mode = None
		self.cnt = None
		self.laser_range = []
		
		
	def modemsg(self,d):
		self.mode = d.mode
		self.cnt = d.cnt
		#print(self.mode)
	
	def callback(self, data):
		self.laser_range = data.ranges
		#print(self.laser_range)
	
	
	def parking(self, a):
		self.turn(a)
		self.sleep(0.33333)
		self.gos(1)
		self.sleep(0.33)
		self.stop()
		self.sleep(0.5)
		self.gos(-1)
		self.sleep(0.33)
		self.turn(-a)
		self.sleep(0.3333)
		self.stop()
		self.sleep(0.5)
		self.park_end.park = 1
		self.sleep(1)
		self.pubpark.publish(self.park_end)
		self.sleep(1)
		
	def sleep(self, a):
		
		b = a
		rate = rospy.Rate(b)
		rate.sleep()
				
		
	def gos(self,a):
		self.vel_tw.linear.x =a*0.1
		self.vel_tw.angular.z = 0
		self.twistpub.publish(self.vel_tw)

	def turn(self, a):
		self.vel_tw.linear.x = 0
		self.vel_tw.angular.z = a * (np.pi)/6
		self.twistpub.publish(self.vel_tw)
		#rate = rospy.Rate(1)
		#rate.sleep()

	def stop(self):
		self.vel_tw.linear.x = 0
		self.vel_tw.angular.z = 0
		self.twistpub.publish(self.vel_tw)
		
		
		
	def lasercut(self):
		print("data")
		self.park_end.park = 0
		
		print("start")
		#self.gos(1.5)
	
		#self.sleep(0.3)
		self.stop()
		self.sleep(0.2)
		
		a = 1
		if self.laser_range[270] >= 0.02 and self.laser_range[270] <= 0.3:
			a = -1
			self.stop()
			self.sleep(1)
			self.gos(1)
			self.sleep(0.333)
		else :
			a =-1
		self.parking(a)
		self.gos(1.2)
		self.sleep(0.2)
		self.stop()
		self.sleep(0.1)
			
def main():
	rospy.init_node("park")
	park = Park()
	
	rate = rospy.Rate(10)
	
	
	while not rospy.is_shutdown():
		if park.mode == 3:
			park.lasercut()
		
		
		rate.sleep()
	
if __name__ == '__main__':
	try:
		main()
	except rospy.ROSInterruptException:
		pass
		
