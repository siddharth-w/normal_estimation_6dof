from numpy import linspace
from secrets import choice
import pybullet as p
import time
import numpy as np
import matplotlib.pyplot as plt
from manipulator_TDC_sid_wrench import Manipulator
import csv

class ArmSim():
    def __init__(self):
        self.robot = Manipulator([0.5,0.4,0.8],[0.5,0.4,0.8])
        self.ee_pos_data = []
        self.timearray, self.t_step= np.linspace(0,25,2500,retstep=True)
        self.torque_data = np.empty([np.size(self.timearray),len(self.robot.rev)])
        self.alpha_in = [ 0 for i in range(len(self.robot.rev))]
        self.h = 1/101
        self.torque_h = np.array([0,0,0,0,0,0])
        self.torque = np.array([0,0,0,0,0,0]) 
        self.t_i = time.time()
        self.t_iter = self.t_i
        self.t_prev = self.t_i
        self.w_sum = 0
        self.positionerror_prev = [0,0,0,0,0,0]
        self.matrix = [0,0,0,0,0,0,0,0,0,0,0,0]
        self.Phi = [0,0,0,0,0,0,0,0,0,0,0,0]
        self.Pi = np.random.rand(12, 12)  # Example: random 6x6 matrix (should be symmetric)
        self.Pi = self.Pi + self.Pi.T  # Ensure P is symmetric
        while not np.all(np.linalg.eigvals(self.Pi) > 0):  # Check if all eigenvalues are positive
            self.Pi = self.Pi + np.eye(12)
        print(np.linalg.eigvals(self.Pi))



        
    def runSim(self):
        self.robot.setInitialState()
        self.robot.turnOffActuators()
        self.robot.turnOffDamping()
        p.enableJointForceTorqueSensor(self.robot.arm,7,enableSensor=1)
        for i,t in zip(range(np.size(self.timearray)),self.timearray):
            # self.robot.axiscreator(self.robot.planeID, 7)
            p.stepSimulation()
            time.sleep(self.h)
            self.torque_h = self.torque
            self.torque = self.get_torque(t,i)
            self.torque_data[i] = self.torque
            self.ee_pos_data.append(list(self.robot.forces))
            p.setJointMotorControlArray(self.robot.arm,self.robot.rev,controlMode=p.TORQUE_CONTROL,forces=self.torque)
            if (time.time() - self.t_i > 60):
                break

    
    def positionError(self):
        self.positionerror =  self.x_des - np.array(self.robot.eef_pos)
        return self.positionerror
    
    def velocityError(self):
        self.velocityerror = self.xd_des - np.array(self.robot.eef_vel)
        return self.velocityerror

    # def write(self, filename):
    #     while self.t<25:
    #         self.t += self.t_step
    #     with open(filename, 'a', newline='') as csvfile:
    #         writer = csv.writer(csvfile, delimiter = ',')
    #         #writer.writeheader()
    #         #writer.writerow(["Time", "x", "y", "z","a","b","c"])
    #         writer.writerow([self.t, self.positionerror[0],self.positionerror[1],self.positionerror[2],self.positionerror[3],self.positionerror[4],self.positionerror[5]])


    def write(self, filename):
        # while self.t<25:
        #     self.t += self.t_step
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')
            #writer.writeheader()
            #writer.writerow(["Time", "x", "y", "z","a","b","c"])
            writer.writerow([time.time() - self.t_i, self.positionerror[0],self.positionerror[1],self.positionerror[2],self.positionerror[3], self.positionerror[4],self.positionerror[5]])
            # print(self.t - time.time())
            # print("F :",self.contactF )

        
    def writeF(self, filename):
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')
            #writer.writeheader()
            #writer.writerow(["Time", "x", "y", "z","a","b","c"])
            writer.writerow([time.time() - self.t_i, self.contactF[0], self.contactF[1], self.contactF[2]])
            # print(self.t - time.time())
            # print("F :",self.contactF )

    def writeE(self, filename):
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')
            self.t_iter = time.time()
            
            B_top = np.zeros((6, 6))
            B_bottom = np.eye(6)
            B = np.vstack((B_top, B_bottom))

            self.Phi[:5] = [self.positionerror[0],self.positionerror[1],self.positionerror[2],self.positionerror[3], self.positionerror[4],self.positionerror[5]]
            self.Phi[6:]= (np.subtract(self.positionerror, self.positionerror_prev))/(self.t_iter - self.t_prev)


            self.w_sum += np.transpose( B.T @ self.Pi @ self.Phi) @ self.contactF
            # print(self.Pi )
            writer.writerow([time.time() - self.t_i, self.w_sum])
            self.t_prev = self.t_iter
            self.positionerror_prev = self.positionerror

    def writeN(self, filename):
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')
            self.t_iter = time.time()
            print(self.positionerror, self.positionerror_prev)

            self.matrix[:5] = [self.positionerror[0],self.positionerror[1],self.positionerror[2],self.positionerror[3], self.positionerror[4],self.positionerror[5]]
            self.matrix[6:]= (np.subtract(self.positionerror, self.positionerror_prev))/(self.t_iter - self.t_prev)

            matrix = np.subtract(self.matrix,[0.00643598, -0.0106456 , -0.29815721,  0.07981045, -0.01439952, 0.18028984, 0,0,0,0,0,0 ])
            writer.writerow([time.time() - self.t_i, matrix.T @ (self.P @ matrix)
                            ])
            self.t_prev = self.t_iter
            self.positionerror_prev = self.positionerror

    def get_torque(self,t,i):
        self.robot.joint_state_info()
        # print("initial", self.robot.eef_pos)
        self.robot.getjacobian()
        self.m = self.robot.massMatrix()

        q_ddot_h = (np.array(self.robot.omega_prev) - np.array(self.robot.omega_pprev))/self.h
        self.M_bar = np.array([[ 9.25958840e-01 ,-3.44950225e-01 , 5.07159378e-02 ,-3.04338478e-04,
                             3.28488058e-04, -2.17260720e-04] ,[-3.44950225e-01,  1.76554356e+00 , 3.04762975e-01 , 1.33003450e-02, 2.34989579e-07 , 8.67767088e-07],
                             [ 5.07159378e-02,  3.04762975e-01 , 6.10282299e-01 , 1.02895488e-02 ,2.34990129e-07,  8.67767088e-07],
                             [-3.04338478e-04 , 1.33003450e-02 , 1.02895488e-02 , 5.85027636e-03 ,2.34990939e-07 , 8.67767088e-07],
                             [ 3.28488058e-04  ,2.34989579e-07 , 2.34990129e-07 , 2.34990939e-07 ,2.08277673e-03 , 6.39887126e-25],
                             [-2.17260720e-04,  8.67767088e-07,  8.67767088e-07 , 8.67767088e-07, 6.39887126e-25 , 2.20016010e-04]])


        #Impedance Control
        self.M = np.diag(np.array([1e0,1e0,1e0,1e0,1e0,1e0]))
        self.Kp = np.diag([120,120,250,120,120,120])
        self.Kd = np.diag(6*[2*np.sqrt(180)])

        #Desired Reference
        self.x_des = np.array([0.4,0.3,0.6,0.1,1.4,0.403])
        self.xd_des = np.array([0,0,0,0,0,0])
        self.xdd_des = np.array([0,0,0,0,0,0])


        self.P = np.zeros((12,12))
        self.P[:6, :6] = self.Kp
        self.P[6:, 6:] = np.eye(6)

        p.addUserDebugLine([0,0,0], self.robot.eef_pos[:3],lineColorRGB=[157/256, 51/256, 0/256],lineWidth=3,lifeTime = 30*self.h)
        p.addUserDebugLine([0,0,0], self.x_des[:3],lineColorRGB=[1,0,1],lineWidth=3,lifeTime = 30*self.h)
        p.addUserDebugLine(self.robot.eef_pos[:3], self.x_des[:3],lineColorRGB=[0,0,1],lineWidth=3,lifeTime = 30*self.h)

        p.addUserDebugLine([0.8,0.4,0.77],self.robot.eef_pos[:3],lineColorRGB=[0,0,0],lineWidth=3,lifeTime = 30*self.h)
        p.addUserDebugLine(np.add(self.robot.table_pos[0], [-0.0,-0.0,-0.73]),self.robot.eef_pos[:3],lineColorRGB=[1,1,0],lineWidth=3,lifeTime = 30*self.h)
        p.addUserDebugLine(np.add(self.robot.table_pos[0], [-0.0,-0.0,-0.73]),[0.8,0.4,0.79],lineColorRGB=[0,1,1],lineWidth=3,lifeTime = 30*self.h)

        #eef axis

        p.addUserDebugLine(lineFromXYZ          = self.robot.eef_pos[:3] ,
                                                                        lineToXYZ = np.add(self.robot.eef_pos[:3] , [0.1,0,0])   ,
                                                                        lineColorRGB = [1, 0, 0] ,
                                                                        lineWidth = 3 ,
                                                                        lifeTime = 30*self.h )
                                                                        # parentObjectUniqueId = planeID ,
                                                                        # parentLinkIndex = linkId )
        p.addUserDebugLine(lineFromXYZ          = self.robot.eef_pos[:3]  ,
                                                                lineToXYZ = np.add(self.robot.eef_pos[:3] , [0,0.1,0])  ,
                                                                lineColorRGB         = [0, 1, 0]  ,
                                                                lineWidth            = 3        ,
                                                                lifeTime             = 30*self.h          )
                                                                # parentObjectUniqueId = planeID     ,
                                                                # parentLinkIndex      = linkId     )

        p.addUserDebugLine(lineFromXYZ          = self.robot.eef_pos[:3]  ,
                                                                lineToXYZ =  np.add(self.robot.eef_pos[:3] , [0.02,0.02,0.1])  ,
                                                                lineColorRGB         = [0, 0, 1]  ,
                                                                lineWidth            = 3        ,
                                                                lifeTime             = 30*self.h         )
                                                                # parentObjectUniqueId = planeID     ,
                                                                # parentLinkIndex      = linkId     )

        #Table Axis

        p.addUserDebugLine(lineFromXYZ          = [0.8,0.4,0.77] ,
                                                                        lineToXYZ = np.add([0.8,0.4,0.77] , [0.1,0,0]),
                                                                        lineColorRGB = [1, 0, 0] ,
                                                                        lineWidth = 3 ,
                                                                        lifeTime = 30*self.h )
                                                                        # parentObjectUniqueId = planeID ,
                                                                        # parentLinkIndex = linkId )
        p.addUserDebugLine(lineFromXYZ          = [0.8,0.4,0.77],
                                                                lineToXYZ = np.add([0.8,0.4,0.77], [0,0.1,0]),
                                                                lineColorRGB         = [0, 1, 0]  ,
                                                                lineWidth            = 3        ,
                                                                lifeTime             = 30*self.h          )
                                                                # parentObjectUniqueId = planeID     ,
                                                                # parentLinkIndex      = linkId     )

        p.addUserDebugLine(lineFromXYZ          = [0.8,0.4,0.77],
                                                                lineToXYZ = np.add([0.8,0.4,0.77], [0,0,0.1]) ,
                                                                lineColorRGB         = [0, 0, 1]  ,
                                                                lineWidth            = 3        ,
                                                                lifeTime             = 30*self.h         )
                                                                # parentObjectUniqueId = planeID     ,
                                                                # parentLinkIndex      = linkId     )


        #table_dynamic

     
        p.addUserDebugText("Frame A", np.add(self.robot.eef_pos[:3], [-0.1,0.12,0]), [0,0,0], 1,30*self.h )
        p.addUserDebugText("Frame B", np.add([0.8,0.4,0.77], [0.1,0.1,0]), [91/256, 1/256, 0], 1,30*self.h )
        p.addUserDebugText("Frame C", np.add(self.robot.table_pos[0], [-0.0,-0.0,-0.73]), [5/256,52/256,108/256], 1,30*self.h )
        p.addUserDebugText("Desired Reference", np.add(self.x_des[:3], [-0.0,-0.0,-0.05]), [0,0,0], 1,30*self.h )




        p.addUserDebugLine(lineFromXYZ = np.add(self.robot.table_pos[0], [-0.0,-0.0,-0.73]),lineToXYZ = np.add(self.robot.table_pos[0], [-0.0,-0.0,-0.63])  ,lineColorRGB=[0,0,1],lineWidth=3,lifeTime = 30*self.h)
        p.addUserDebugLine(lineFromXYZ = np.add(self.robot.table_pos[0], [-0.0,-0.0,-0.73]),lineToXYZ = np.add(self.robot.table_pos[0], [-0.0,0.1,-0.73]) ,lineColorRGB=[0,1,0],lineWidth=3,lifeTime = 30*self.h)
        p.addUserDebugLine(lineFromXYZ = np.add(self.robot.table_pos[0], [-0.0,-0.0,-0.73]),lineToXYZ = np.add(self.robot.table_pos[0], [0.1,-0.0,-0.73]),lineColorRGB=[1,0,0],lineWidth=3,lifeTime = 30*self.h)




        self.contactF = self.robot.contactForce()
        self.alpha_in = self.xdd_des + np.linalg.inv(self.M)@(self.Kp@self.positionError() + self.Kd@self.velocityError()+self.robot.contactForce())
        self.ah = np.linalg.inv(self.robot.analyticjacobian())@(self.alpha_in - self.robot.Jdot@self.robot.omega)
        # print(self.positionError())
        self.write("position_Error_d.csv")
        self.writeF("force_Error_d.csv")
        # self.writeE("E.csv")
        # self.writeN("N.csv")
        return self.torque_h - self.M_bar@q_ddot_h + self.M_bar@self.ah 
        
    

if __name__ == "__main__":
    hl, = plt.plot([], [])
    r1 = ArmSim()
    r1.runSim()
