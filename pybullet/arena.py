import pybullet as pb
import pybullet_data
import numpy as np

class Arena:
    def __init__(self, num_agents = 10, fps = 60.0, gui = True):
        self.num_agents = num_agents
        
        if gui:
            pb.connect(pb.GUI)
        else: 
            pb.connect(pb.DIRECT)

        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, -9.8)

        self.dt = 1.0 / fps
        pb.setTimeStep(self.dt)

        self.plane = pb.loadURDF("plane.URDF")

        self.create_walls()
        self.agent_ids = self.create_agents()
        self.debug_lines = [-1] * self.num_agents

    #arena world setup
    def create_walls(self):
        wall_thickness = 0.1
        wall_height = 2.0
        size = 5

        #top/bottom walls
        col_shape = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size, wall_thickness, wall_height])

        pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape, basePosition=[0, size, wall_height])
        pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape, basePosition=[0, -size, wall_height])

        #side walls
        col_shape_side = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[wall_thickness, size, wall_height])

        pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape_side, basePosition=[-size, 0, wall_height])
        pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape_side, basePosition=[size, 0, wall_height])

    def create_agents(self):
        agent_ids = []

        self.radius = 0.15
        self.height = 0.2
        mass = 1.0

        col_shape = pb.createCollisionShape(pb.GEOM_CYLINDER, radius=self.radius, height=self.height)
        visual_shape = pb.createVisualShape(pb.GEOM_CYLINDER, radius=self.radius, length=self.height, rgbaColor=[1, 0.2, 0.2, 1])

        for jawn in range(self.num_agents):
            pos = np.random.uniform(-2, 2, size=2)
            
            body = pb.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col_shape, baseVisualShapeIndex=visual_shape, 
                                     basePosition=[pos[0], pos[1], self.height / 2])
            pb.changeDynamics(body, -1, angularDamping=1.0, linearDamping=0.1)

            agent_ids.append(body)

        return agent_ids
    
    def step(self):
        pb.stepSimulation()
        self.draw_headings()

    def get_states(self):
        positions = []
        velocities = []

        for id in self.agent_ids:
            pos, _ = pb.getBasePositionAndOrientation(id)
            vel, _ = pb.getBaseVelocity(id)

            positions.append([pos[0], pos[1]])
            velocities.append([vel[0], vel[1]])

        return np.array(positions), np.array(velocities)
    
    def apply_actions(self, desired_velocities):
        for id, v in zip(self.agent_ids, desired_velocities):
            pb.resetBaseVelocity(id, linearVelocity=[v[0], v[1], 0])

    def draw_headings(self):
        for i, id in enumerate(self.agent_ids):
            pos, _ = pb.getBasePositionAndOrientation(id)
            vel, _ = pb.getBaseVelocity(id)

            v = np.array([vel[0], vel[1]])
            speed = np.linalg.norm(v) 

            if speed < 1e-4:
                dir = np.array([1.0, 0.0])
            else:
                dir = v / speed
            
            start = np.array([pos[0], pos[1], pos[2] + 0.1])
            end = start + np.array([dir[0], dir[1], 0]) * 0.5

            self.debug_lines[i] = pb.addUserDebugLine(start, end, [0, 1, 0], lineWidth=2, 
                                                      replaceItemUniqueId=self.debug_lines[i])
            
    def reset(self):
        for id in self.agent_ids:
            pos = np.random.uniform(-2, 2, size=2)

            pb.resetBasePositionAndOrientation(id, [pos[0], pos[1], self.height / 2], [0, 0, 0, 1])

            vx, vy = np.random.uniform(-0.5, 0.5, size=2)
            pb.resetBaseVelocity(id, [vx, vy, 0], [0, 0, 0])