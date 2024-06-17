import pybullet as p
import time
import pybullet_data
import numpy as np


class Manipulator:
    def __init__(self, intpt, fipt):


        self.physicsClient = p.connect(p.GUI)
        p.setGravity(0,0,-10)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setRealTimeSimulation(0)
        p.setTimeStep(1/240)



        self.planeID = p.loadURDF("plane.urdf")
        self.arm_urdfpath = "models/ur5-robotiq-140/urdf/ur5_robotiq_140.urdf"
        self.arm = p.loadURDF(self.arm_urdfpath,useFixedBase= True,basePosition=[0,0,0])
        self.tablepath = "models/ur5-robotiq-140/urdf/objects/table.urdf"
        self.table = p.loadURDF(self.tablepath, basePosition=[0.8,0.4,0.5])
        # self.table = p.loadURDF(self.tablepath, basePosition=[1,1,0])


        # self.table = p.loadURDF(self.tablepath,useFixedBase=True,basePosition=[0.5,0.4,0.7],baseOrientation = [0.01,0.1,0.2,-0.5])
        p.enableJointForceTorqueSensor(self.arm, 8, True)
        cid = p.createConstraint(self.table, -1, -1, -1, p.JOINT_FIXED,[0, 0, 0],[0, 0, 0],  [0.8, 0.4, 1.5])
        # cid = p.createConstraint(self.table, -1, -1, -1, p.JOINT_FIXED,[0, 0, 0],[0, 0, 0],  [1,1,1])

        # print(cid)
        p.changeConstraint(cid, maxForce=20)



        # self.planeID = p.loadURDF("plane.urdf")
        # self.arm_urdfpath = "urdf/ur5-robotiq-140/urdf/ur5_robotiq_140.urdf"
        # self.arm = p.loadURDF(self.arm_urdfpath,useFixedBase= True,basePosition=[0,0,0])

        # self.tablepath = "urdf/ur5-robotiq-140/urdf/objects/table.urdf"
        # self.table = p.loadURDF(self.tablepath,useFixedBase=False, basePosition=[0.5,0.4,0.7])
        # # self.table = p.loadURDF(self.tablepath,useFixedBase=True,basePosition=[0.5,0.4,0.7],baseOrientation = [0.01,0.1,0.2,-0.5])
        # p.enableJointForceTorqueSensor(self.arm, 8, True)
        # cid = p.createConstraint(self.table, -1, -1, -1, p.JOINT_FIXED,[0, 0, 0],[0, 0, 0], [0.7, 0.25, 1.8])
        # print(cid)
        # p.changeConstraint(cid, maxForce=20)

        self.noJoints = p.getNumJoints(self.arm)
        '''intpt = [ intial coordinate, intial velocity, intial acceleration] of end effector'''
        self.intialPt = np.array(list(intpt) + list([0,1.4,0.203]))
        '''fipt = [ final coordianate, final velocity, final acceleration] of end effector'''
        self.finalPt = np.array(list(fipt) + list([0.1,1.4,0.203]))
        self.revjoints()
        self.no_j = p.getNumJoints(self.arm)
        self.rest = []
        self.lowerlimits = []
        self.upperlimits = []
        self.ranges = []
        for i in self.rev:
            info = p.getJointInfo(self.arm, i)
            # Retrieve lower and upper ranges for each relevant joint
            if info[3] > -1:
                self.rest.append(0)
                self.lowerlimits.append(info[8])
                self.upperlimits.append(info[9])
                self.ranges.append(info[9] - info[8])
        self.omega = [0, 0, 0, 0, 0, 0]
        self.omega_prev = [0, 0, 0, 0, 0, 0]
        self.angles = [0, 0, 0, 0, 0, 0]
        self.angles_prev = [0, 0, 0, 0, 0, 0]
        self.angles_pprev = [0, 0, 0, 0, 0, 0]
        self.eef_vel = [0, 0, 0, 0, 0, 0]
        self.eef_vel_p = [ 0, 0, 0, 0, 0, 0]
        self.geometricJacobian = np.zeros((6,6))
        self.Ja_prev = np.zeros((6,6))
        p.setRealTimeSimulation(0)
    
    def joint_state_info(self):
        joints = p.getJointStates(self.arm,self.rev) 
        # print((p.getJointStates(self.arm, [3])))
        # print((p.getJointState(self.arm, 3)))

        self.omega_pprev = self.omega_prev
        self.omega_prev = self.omega
        self.angles_pprev = self.angles_prev
        self.angles_prev = self.angles
        self.angles = [ i[0] for i in joints]
        self.omega = [ i[1] for i in joints]
        self.forces = np.array(p.getJointState(self.arm,8)[2])
        # self.forces = np.concatenate((self.forces[:3],[0,0,0]))
        if(np.linalg.norm(self.forces[:3]) > 0.001):
            #print(self.forces)
            pass
        
        self.getjacobian()
        # print(self.angles)
        self.eef_vel_pp = self.eef_vel_p
        self.eef_vel_p = self.eef_vel
        self.eef_pos = np.array(list(p.getLinkState(self.arm,8,computeLinkVelocity=True)[0])+list(p.getEulerFromQuaternion(p.getLinkState(self.arm,7,computeLinkVelocity=True)[1]))) 


        self.table_pos = p.getBasePositionAndOrientation(self.table)
        # self.table_pos = np.array(list(p.getBasePositionAndOrientation(self.table)[0])+list(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.table)[1])))
        self.rot_eef = self.quaternion_rotation_matrix(p.getLinkState(self.arm,7,computeLinkVelocity=True)[1])
        self.rot_eef1 = p.getMatrixFromQuaternion(p.getLinkState(self.arm,7,computeLinkVelocity=True)[1])

        self.rot_table = self.quaternion_rotation_matrix(self.table_pos[1])                                 
        self.rot_table1 = p.getMatrixFromQuaternion(self.table_pos[1])                                 

        self.eef_vel = np.array(self.analyticjacobian())@np.array(self.omega)
        # self.alpha_curr = self.Jdot@self.omega +self.analyticjacobian()@((np.array(self.omega) - np.array(self.omega_prev))/0.0099)
        # print("Velocity", self.eef_vel)
        # print("Velocity", self.omega)

        self.alpha_curr = (np.array(self.omega_prev) - np.array(self.omega_pprev))/0.0099
 
    def quaternion_rotation_matrix(self,Q):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.
    
        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
    
        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix. 
                This rotation matrix converts a point in the local reference 
                frame to a point in the global reference frame.
        """
        # Extract the values from Q
        q0 = Q[1]
        q1 = Q[2]
        q2 = Q[3]
        q3 = Q[0]
        
        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        
        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        
        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
        
        # 3x3 rotation matrix
        self.rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])
                                
        return self.rot_matrix



    def revjoints(self):
        self.rev = []
        for i in range(8):
            jointinfo = p.getJointInfo(self.arm,i)
            if jointinfo[2] == 0:
                self.rev.append(i)
            if jointinfo[1] == b'ee_fixed_joint':
                self.endEffectorIndex = i 
        
        self.zero_array = [0 for _ in range(len(self.rev))]

    def getjacobian(self):
        endEffector = p.getLinkState(self.arm,8)
        j_prev = self.Ja_prev
        self.linJac,angJac = p.calculateJacobian(self.arm,7,endEffector[2],self.angles,self.zero_array,self.zero_array)
        self.geometricJacobian = np.concatenate((self.linJac, angJac), axis=0)
        
        self.Jdot = (np.array(j_prev) - np.array(self.analyticjacobian()))/0.0099
        self.Ja_prev = self.analyticjacobian()
        return self.geometricJacobian
    
    def analyticjacobian(self):
        endEffector = p.getLinkState(self.arm,7)
        x,y,z = p.getEulerFromQuaternion(endEffector[5])
        B = np.array([
            [1,           0,            np.sin(y)],
            [0,   np.cos(x), -np.cos(y)*np.sin(x)],
            [0,   np.sin(x),  np.cos(x)*np.cos(y)]
        ])
        B_inv = np.linalg.inv(B)
        Ee_M = np.block([
                    [np.eye(3,3),     np.zeros((3,3))],
                    [np.zeros((3,3)),           B_inv]
        ])
        self.Ja = Ee_M@self.geometricJacobian
        return self.Ja
    

    def wrench(self):
        ft = np.array(p.getJointState(self.arm,7)[2])
        f = p.getContactPoints(self.arm,self.table,8,-1)
        wrench = np.array([0.0,0.0,0.0,0.0,0.0,0])
        if len(f) > 0:
            # print(np.array([f[0][10],f[0][12],f[0][9]]))
            for i in range(len(f)):
                wrench += np.array([f[i][10],f[i][12],f[i][9],0,0,0])
        wrench[3] = ft[3]
        wrench[4] = ft[4]
        wrench[5] = ft[5]
        return wrench


    def contactForce(self):
        # print("force-torque ",self.ft)
        f = p.getContactPoints(self.arm,self.table,8,-1)
        contact_f = np.array([0.0,0.0,0.0,0.0,0.0,0])
        if len(f) > 0:
            # print(np.array([f[0][10],f[0][12],f[0][9]]))
            for i in range(len(f)):
                contact_f += np.array([f[i][10],f[i][12],f[i][9],0,0,0])
                # print(contact_f)
        return contact_f

    def getTraj(self,t):
        ipnt = self.intialPt + t*(self.finalPt - self.intialPt)
        xd_des = self.finalPt - self.intialPt
        return ipnt,xd_des

    def setTrajPt(self,p1,p2,p3,p4):
        self.p1,self.p2,self.p3,self.p4 = p1,p2,p3,p4
        
    def get_B(self):
        A = self.getA(0,1,2,3)
        self.B = np.zeros((6,12))
        intialPt = self.finalPt - self.intialPt
        for i in range(len(self.p1)):
            C = np.array([self.p1[i],0,0,self.p2[i],self.p2[i],0,self.p3[i],self.p3[i],0,self.p4[i],0,0])
            self.B[i] = np.linalg.pinv(A)@C

    def getrunningTraj(self,t):
        z = np.zeros((4))
        if t <= 1:
            T = np.concatenate((np.array([1,t,t**2,t**3]),z,z))
            Tdot = np.concatenate((np.array([0,1, 2*t, 3*t**2]),z,z))
            Tddot = np.concatenate((np.array([0,0,  2,   6*t]),z,z))
        elif 1 < t <=2:
            T = np.concatenate((z,np.array([1,t,t**2,t**3]),z))
            Tdot = np.concatenate((z,np.array([0,1, 2*t, 3*t**2]),z))
            Tddot = np.concatenate((z,np.array([0,0,  2,   6*t]),z))
        elif 2 < t <= 3:
            T = np.concatenate((z,z,np.array([1,t,t**2,t**3])))
            Tdot = np.concatenate((z,z,np.array([0,1, 2*t, 3*t**2])))
            Tddot = np.concatenate((z,z,np.array([0,0,  2,   6*t])))
        return self.B@T, self.B@Tdot, self.B@Tddot

    def getA(self,t_0,t_1,t_2,t_f):
        f = lambda t: np.array([1,t,t**2,t**3])
        fdot = lambda t: np.array([0,1, 2*t, 3*t**2])
        fddot = lambda t: np.array([0,0,  2,   6*t])
        z = np.zeros((4))
        A = np.array([  np.concatenate((f(t_0),                   z,         z)),
                        np.concatenate([fdot(t_0),                z,         z]),
                        np.concatenate([fddot(t_0),               z,         z]),
                        np.concatenate((f(t_1),                   z,         z)),
                        np.concatenate([z,                   f(t_1),         z]),
                        np.concatenate([-1*fdot(t_1),     fdot(t_1),         z]),
                        np.concatenate([z,                   f(t_2),         z]),
                        np.concatenate([z,                        z,    f(t_2)]),
                        np.concatenate([z,             -1*fdot(t_2), fdot(t_2)]),
                        np.concatenate([z,                        z,    f(t_f)]),
                        np.concatenate([z,                        z, fdot(t_f)]),
                        np.concatenate([z,                        z,fddot(t_f)])])
        return A


    def massMatrix(self):
        self.getjacobian()
        massmatrix = np.array(p.calculateMassMatrix(self.arm,self.angles))
        # massmatrix = (np.linalg.pinv(self.linJac).T@massmatrix)@np.linalg.pinv(self.linJac)
        return massmatrix
    
    def coriolisVector(self):
        # self.joint_state_info()
        coriolisgravityvector = np.array(p.calculateInverseDynamics(self.arm,self.angles,self.omega,self.zero_array))
        coriolis = coriolisgravityvector - self.gravityVector()
        
        self.getjacobian()
        # coriolis = (np.linalg.pinv(self.linJac).T@coriolis) - self.massMatrix("i")@self.Jdot@self.omega
        return coriolis

    def gravityVector(self):
        # self.joint_state_info()
        gravityvector = p.calculateInverseDynamics(self.arm,objPositions = self.angles,objVelocities = self.zero_array,objAccelerations = self.zero_array)
        gravityvector = np.array(gravityvector)
        return gravityvector

    def setInitialState(self):
        self.upperlimits[1] = -1.5
        intialangles = p.calculateInverseKinematics(self.arm,7,targetPosition=self.intialPt[0:3],targetOrientation =  p.getQuaternionFromEuler(self.intialPt[3:7]),lowerLimits= self.lowerlimits,upperLimits= self.upperlimits,restPoses= self.rest, jointRanges= self.ranges)
        # intialangles = [0,0,0,0,0,0]
        for _ in range(60):
            p.stepSimulation()
            p.setJointMotorControlArray(self.arm,self.rev,p.POSITION_CONTROL,targetPositions = intialangles)
            # print(p.getLinkState(self.arm,7)[0])
            time.sleep(1/100)
    
    def turnOffActuators(self):
        p.setJointMotorControlArray(self.arm,self.rev,p.VELOCITY_CONTROL,forces = self.zero_array)
    
    def turnOffDamping(self):
        for i in self.rev:
            p.changeDynamics(self.arm, i, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
            # p.changeDynamics(self.arm, i, maxJointVelocity=200)
    


    # def axiscreator(self,planeID, linkId = -1):
    #     print(f'axis creator at planeID = {planeID} and linkId = {linkId} as XYZ->RGB')
        
    #     return [x_axis, y_axis, z_axis]