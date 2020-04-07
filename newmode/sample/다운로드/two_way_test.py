#!/usr/bin/env python
import rospy
import numpy as np
import cv2
from newmode.msg import msg_lane
from newmode.msg import msg_sign

class two_way:
	def __init__(self):
		self.x1_ye = None
		self.x2_ye = None
		self.y1_ye = None
		self.y2_ye = None
		self.x1_wh = None
		self.x2_wh = None
		self.y1_wh = None
		self.y2_wh = None
		self.angle = None
		self.msg_lane = msg_lane()
		self.msg_sign = msg_sign()
		self.lane_msg_sub = rospy.Subscriber('/lane_msg', self.msg_lane, self.callback_lane)
		self.sign_msg_sub = rospy.Subscriber('/sine_msg', self.msg_sign, self.callback_sign)
		self.lane_msg_pub = rospy.Publisher('/lane_msg', self.msg_lane, queue_size=1)


	def callback_lane(self, msg):
		self.x1_ye = msg.x1_ye
		self.x2_ye = msg.x2_ye
		self.y1_ye = msg.y1_ye
		self.y2_ye = msg.y2_ye
		self.x1_wh = msg.x1_wh
		self.x2_wh = msg.x2_wh
		self.y1_wh = msg.y1_wh
		self.y2_wh = msg.y2_wh

	def callback_sign(self, msg):
		self.sign = msg.name

	def angle_cul(self):
		if self.sign == "RIGHT SIGN" or self.sign == "LEFT SIGN" or self.sign == "CANTGO SIGN":
			if self.sign == "RIGHT SIGN":
				self.x1_ye = msg.x1_ye
				self.y1_ye = msg.y1_ye
				self.x2_ye = msg.x2_ye
				self.y2_ye = msg.y2_ye
				self.angle = np.arctan2((y1 - y2), (x1 - x2)) * (180 / np.pi)

			if self.sign == "LEFT SIGN":
				self.x1_wh = msg.x1_wh
				self.y1_wh = msg.y1_wh
				self.x2_wh = msg.x2_wh
				self.y2_wh = msg.y2_wh
				self.angle = np.arctan2((y1 - y2), (x1 - x2)) * (180 / np.pi)

			if self.sign == "CANTGO SIGN":
				if self.sign_1 == "RIGHT SIGN":
					self.angle = np.arctan2((y1 - y2), (x1 - x2)) * (180 / np.pi)
						#msg_lane.angle = angle

				if self.sign_1 == "LEFT SIGN":
					self.angle = np.arctan2((y1 - y2), (x1 - x2)) * (180 / np.pi)

			if self.angle <= 180 and self.angle >= -180:
				if self.angle > 70 and self.angle < 110:
					self.angle = 90 - self.angle
    
				else:
					self.angle = self.angle / 2

			self.msg_lane.angle = self.angle
			rospy.loginfo("%f" % (self.angle))
			self.lane_msg_pub.publish(self.msg_lane)

			if self.sign is not None and self.sign is not "CANTGO SIGN":
				self.sign_1 = self.sign


def main():
	rospy.init_node("two_way")
	rate = rospy.Rate(10)
	tw = two_way()
	tw.angle_cul()


if __name__ == '__main__':
	try:
		main()
	except rospy.ROSInterruptException:
		pass




