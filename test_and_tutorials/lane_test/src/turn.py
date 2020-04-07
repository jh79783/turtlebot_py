#!/usr/bin/env python
import rospy
from fm.msg import twist
from lane_test.msg import msg_lane
PI = 3.141592
def callback(data):
    print("ddd")
    rospy.loginfo('scan.ranges[359]: %s',data.angle)
    if data.angle >0:
        new_angle = 90-data.angle
    elif data.angle <0:
        new_angle = -90-data.angle
    elif data.angle == 0 :
        new_angle = 1  
    print("",new_angle)
    gos(new_angle)


    velocity_publisher.publish(vel_msg)
    
    
def gos(an):
    #Starts a new node
    #rospy.init_node('robot_cleaner', )
    #velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
    #vel_msg = Twist()

    # Receiveing the user's input
    print("Let's rotate your robot")
    speed = an
    angle = 1
    #if an >= 0:
    #    clockwise = True #True or false
    #elif an < 0 :
    #    clockwise = False

    #Converting from angles to radians
    angular_speed = speed*2*PI/360
    relative_angle = angle*2*PI/360

    #We wont use linear components
    vel_msg.linear.x=0.1
    vel_msg.linear.y=0
    vel_msg.linear.z=0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0

    # Checking if our movement is CW or CCW

    vel_msg.angular.z =angular_speed
    # Setting the current time for distance calculus
    t0 = rospy.Time.now().to_sec()
    current_angle = 0

    while(current_angle < relative_angle):
        velocity_publisher.publish(vel_msg)
        t1 = rospy.Time.now().to_sec()
        current_angle = abs(angular_speed*(t1-t0))


    #Forcing our robot to stop
    vel_msg.angular.z = 0
    velocity_publisher.publish(vel_msg)
    
    

def move1():
    while not rospy.is_shutdown():
        rospy.spin()






if __name__ == '__main__':
    rospy.init_node('move_turn', anonymous=True)
    velocity_publisher = rospy.Publisher('/mode_twist', twist, queue_size=10)
    rospy.Subscriber('/msg_lane', msg_lane, callback , queue_size=10)
    vel_msg = twist()
      
  
    try:
        #Testing our function
        while(1):
            move1()

          
    except rospy.ROSInterruptException: pass
