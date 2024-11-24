from math import sin
import numpy as np
import sympy as sp
from .dynamic_model import DynamicModelBase
from utils.Logger import logger
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import math 
from sympy.utilities.lambdify import lambdify

class QuadCopter(DynamicModelBase):
    def __init__(self, is_with_constraints = True, T = 100):
        ##### Dynamic Function ########
        n, m = 12, 4 # number of state = 12, number of action = 4, prediction horizon = 100
        h_constant = 0.02 # sampling time
        x_u_var = sp.symbols('x_u:16') 
        ueq = 1.962
        # p_x p_y p_z
        # v_x v_y v_z
        # phi(6) theta(7) psi(8)
        # omega_x omega_y omega_z
        # f1 f2 f3 f4
        Jx = 0.0244
        Jy = 0.0244
        Jz = 0.0436
        mass = 0.8
        g_constant = 9.81 
        L_constant = 0.165 # m
        c_constant = 0.002167 # m
        cos_phi = sp.cos(x_u_var[6])
        sin_phi = sp.sin(x_u_var[6])
        cos_theta = sp.cos(x_u_var[7])
        sin_theta = sp.sin(x_u_var[7])
        cos_psi = sp.cos(x_u_var[8])
        sin_psi = sp.sin(x_u_var[8])
        
        e_constant = np.asarray([0,0,1]).reshape(-1,1)
        R_matrix = sp.Matrix([[cos_theta*cos_psi, cos_theta*sin_psi, -sin_theta],
                        [sin_phi*sin_theta*cos_psi-cos_phi*sin_psi, sin_phi*sin_theta*sin_psi+cos_phi*cos_psi, sin_phi*cos_theta],
                        [cos_phi*sin_theta*cos_psi+sin_phi*sin_psi, cos_phi*sin_theta*sin_psi-sin_phi*cos_psi, cos_phi*cos_theta]])
        W_matrix = sp.Matrix([[1.0, sin_phi*sin_theta/cos_theta, cos_phi*sin_theta/cos_theta],
                      [0.0, cos_phi, -sin_phi],
                      [0.0, sin_phi/cos_theta, cos_phi/cos_theta]])
        J_matrix = np.diag([Jx, Jy, Jz]) 
        pos = sp.Matrix([[x_u_var[0]], [x_u_var[1]], [x_u_var[2]]])
        vel = sp.Matrix([[x_u_var[3]], [x_u_var[4]], [x_u_var[5]]])
        ang = sp.Matrix([[x_u_var[6]], [x_u_var[7]], [x_u_var[8]]])
        ang_vel = sp.Matrix([[x_u_var[9]], [x_u_var[10]], [x_u_var[11]]])
        # Dynamics params
        pos_dot = R_matrix.T * vel
        vel_dot = -ang_vel.cross(vel) + R_matrix @ (g_constant * e_constant)
        ang_dot = W_matrix * ang_vel
        angvel_dot = np.linalg.inv(J_matrix) @ (-ang_vel.cross(J_matrix * ang_vel))

        # Make constant Bc matrix
        Bc = np.zeros((12, 4))
        Bc[5, 0] = -1.0/mass
        Bc[5, 1] = -1.0/mass
        Bc[5, 2] = -1.0/mass
        Bc[5, 3] = -1.0/mass
        Bc[9, 1] = -L_constant/Jx
        Bc[9, 3] = L_constant/Jx
        Bc[10, 0] = L_constant/Jy
        Bc[10, 2] = -L_constant/Jy
        Bc[11, 0] = -c_constant/Jz
        Bc[11, 1] = c_constant/Jz
        Bc[11, 2] = -c_constant/Jz
        Bc[11, 3] = c_constant/Jz

        dynamic_function = sp.Matrix([
            pos + pos_dot * h_constant, 
            vel + vel_dot * h_constant, 
            ang + ang_dot * h_constant, 
            ang_vel + angvel_dot * h_constant]) + h_constant * Bc * sp.Matrix([[x_u_var[12] + ueq], [x_u_var[13] + ueq], [x_u_var[14] + ueq], [x_u_var[15] + ueq]])
        self.dynamic_function_lambdified = lambdify(x_u_var, dynamic_function, 'numpy')
        init_state = np.asarray([0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.float64).reshape(-1,1) 
        init_action = np.zeros((T,m,1)) 

        if is_with_constraints: 
            box_constr = np.asarray([
                [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], 
                [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf],
                [-np.pi/2, np.pi/2], [-np.pi/2, np.pi/2], [-np.pi, np.pi],
                [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf],
                [-2, 2], [-2, 2], [-2, 2], [-2, 2]]) 
            other_constr =  [-((x_u_var[0] - 0.3)**2 + (x_u_var[1] - 0.3)**2 + (x_u_var[2] - 0.3)**2 - 0.01), 
                            -((x_u_var[0] - 0.5)**2 + (x_u_var[1] - 0.5)**2 + (x_u_var[2] - 0.6)**2 - 0.01),
                            -((x_u_var[0] - 0.7)**2 + (x_u_var[1] - 0.7)**2 + (x_u_var[2] - 0.7)**2 - 0.01)]
        else:
            box_constr = np.asarray([
                [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], 
                [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf],
                [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf],
                [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf],
                [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]) 
            other_constr = []
        ##### Objective Function ########
        position_var = sp.symbols("p:3") # x and y
        add_param = np.hstack([np.ones(T).reshape(-1,1), np.ones(T).reshape(-1,1), np.ones(T).reshape(-1,1)])
        C_matrix = np.diag(np.zeros(16))
        C_matrix[0,0] = C_matrix[1,1] = C_matrix[2,2] = 10
        C_matrix[3,3] = C_matrix[4,4] = C_matrix[5,5] = 1
        r_vector = np.asarray([
            position_var[0], position_var[1], position_var[2],
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.,
            0.,0.,0.,0.])
            
        obj_fun = (x_u_var - r_vector)@C_matrix@(x_u_var - r_vector)
        super().__init__(   dynamic_function=sp.Array(dynamic_function)[:,0], 
                            x_u_var = x_u_var, 
                            box_constr = box_constr, 
                            other_constr = other_constr,
                            init_state = init_state, 
                            init_action = init_action, 
                            obj_fun = obj_fun,
                            add_param_var= position_var,
                            add_param= add_param)


    def rotation_matrix(self,angles):
        ct = math.cos(angles[0])
        cp = math.cos(angles[1])
        cg = math.cos(angles[2])
        st = math.sin(angles[0])
        sp = math.sin(angles[1])
        sg = math.sin(angles[2])
        R_x = np.array([[1,0,0],[0,ct,-st],[0,st,ct]])
        R_y = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        R_z = np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R

    def update(self):
        for key in self.quads:
            R = self.rotation_matrix(self.quads[key]['orientation'])
            L = self.quads[key]['L']
            points = np.array([ [-L,0,0], [L,0,0], [0,-L,0], [0,L,0], [0,0,0], [0,0,0] ]).T
            points = np.dot(R,points)
            points[0,:] += self.quads[key]['position'][0]
            points[1,:] += self.quads[key]['position'][1]
            points[2,:] += self.quads[key]['position'][2]
            self.quads[key]['l1'].set_data(points[0,0:2],points[1,0:2])
            self.quads[key]['l1'].set_3d_properties(points[2,0:2])
            self.quads[key]['l2'].set_data(points[0,2:4],points[1,2:4])
            self.quads[key]['l2'].set_3d_properties(points[2,2:4])
            self.quads[key]['hub'].set_data(points[0,5],points[1,5])
            self.quads[key]['hub'].set_3d_properties(points[2,5])
        plt.pause(0.000000000000001)

    def evaluate_dynamics(self, current_state, output):
        # 상태 및 출력을 결합하여 동적 함수에 입력으로 사용
        x_u = np.hstack((current_state, output))
        # 동적 함수 계산
        next_state = self.dynamic_function_lambdified(*x_u).astype(np.float64).flatten()
        return next_state

    def play(self, logger_folder=None, no_iter = -1):
        """ If logger_folder exists and the result file is saved, then the specific iteration can be chosen to play the animation. \\

            Parameter
            ----------
            logger_folder : string
                The name of the logger folder
            no_iter : int
                The number of iteration to play the animation
        """
        fig, ax = super().create_plot(figsize=(8, 8), xlim=(0,1.05), ylim=(0,1.05), zlim=(0,1.05), is_3d=True, is_equal = False)
        def draw_sphere(xx,yy,zz,rr):
            u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
            x = xx + rr*np.cos(u)*np.sin(v)
            y = yy + rr*np.sin(u)*np.sin(v)
            z = zz + rr*np.cos(v)
            ax.plot_wireframe(x, y, z, color="silver", alpha = 0.6)
        draw_sphere(0.3, 0.3, 0.3, 0.1)
        draw_sphere(0.5, 0.5, 0.6, 0.1)
        draw_sphere(0.7, 0.7, 0.7, 0.1)
        self.quads = {'q1':{'position':[0,0,0],'orientation':[0,0,0],'L':0.1}}
        for key in self.quads:
            self.quads[key]['l1'], = ax.plot([],[],[],color='deepskyblue',linewidth=3,antialiased=False)
            self.quads[key]['l2'], = ax.plot([],[],[],color='skyblue',linewidth=3,antialiased=False)
            self.quads[key]['hub'], = ax.plot([],[],[],marker='o',color='orange', markersize = 10, antialiased=False)
        trajectory = np.asarray(logger.read_from_json(logger_folder, no_iter)["trajectory"])
        ax.plot3D(trajectory[:,0,0], trajectory[:,1,0], trajectory[:,2,0], color = 'lightcoral')

        last_position = trajectory[-1, :3]
        print(f'Last state with no noise: Position = {last_position}')
        
        self._is_interrupted=False
        for i in range(self.T):
            # car.center = trajectory[i,0,0], trajectory[i,1,0]
            self.quads['q1']['position'] = [trajectory[i,0,0], trajectory[i,1,0], trajectory[i,2,0]]
            self.quads['q1']['orientation'] = [trajectory[i,6,0], trajectory[i,7,0], trajectory[i,8,0]]
            self.update()
            if self._is_interrupted:
                return
        self._is_interrupted=False

    def play_with_multiple_noise(self, logger_folder=None, no_iter=-1, noise_sigmas=[0.001, 0.005, 0.01]):
        """ Play the animation with multiple noise levels added to the position.

            Parameter
        ----------
            logger_folder : string
                The name of the logger folder
            no_iter : int
                The number of iteration to play the animation
            noise_sigmas : list of float
                The list of standard deviations of the Gaussian noise to add to the position
        """
        fig, ax = super().create_plot(figsize=(8, 8), xlim=(0, 1.1), ylim=(0, 1.1), zlim=(0, 1.1), is_3d=True, is_equal=False)

        def draw_sphere(xx, yy, zz, rr):
            u, v = np.mgrid[0:2 * np.pi:10j, 0:np.pi:10j]
            x = xx + rr * np.cos(u) * np.sin(v)
            y = yy + rr * np.sin(u) * np.sin(v)
            z = zz + rr * np.cos(v)
            ax.plot_wireframe(x, y, z, color="silver", alpha=0.6)

        draw_sphere(0.3, 0.3, 0.3, 0.1)
        draw_sphere(0.5, 0.5, 0.6, 0.1)
        draw_sphere(0.7, 0.7, 0.7, 0.1)
        self.quads = {'q1': {'position': [0, 0, 0], 'orientation': [0, 0, 0], 'L': 0.1}}
        for key in self.quads:
            self.quads[key]['l1'], = ax.plot([], [], [], color='deepskyblue', linewidth=3, antialiased=False)
            self.quads[key]['l2'], = ax.plot([], [], [], color='skyblue', linewidth=3, antialiased=False)
            self.quads[key]['hub'], = ax.plot([], [], [], marker='o', color='orange', markersize=10, antialiased=False)    

        trajectory = np.asarray(logger.read_from_json(logger_folder, no_iter)["trajectory"])
        ax.plot3D(trajectory[:, 0, 0], trajectory[:, 1, 0], trajectory[:, 2, 0], label='No Noise', color='black')
        ax.legend('No noise')

        #no noise의 경우
        last_position = trajectory[-1, :3]
        print(f'Last state with no noise: Position = {last_position}')

        collision = False
        for sphere_center, sphere_radius in [([0.3, 0.3, 0.3], 0.1), ([0.5, 0.5, 0.6], 0.1), ([0.7, 0.7, 0.7], 0.1)]:
            distance = np.linalg.norm(last_position - np.array(sphere_center))
            if distance < sphere_radius:
                collision = True
                break
        if collision:
            print(f"Collision detected with no noise!")
        else:
            print(f"No collision detected with no noise.")
    
        #noise 있는 경우
        def simulate_with_noise(trajectory, noise_sigma):
            noisy_trajectory = []
            current_state = trajectory[0, :12].flatten()  # 초기 상태

            for t in range(trajectory.shape[0]):
                output = trajectory[t, 12:16].flatten()  # 계획된 output
                next_state = self.evaluate_dynamics(current_state, output)  # 시스템의 다이나믹스 적용
                # Position에 noise 추가
                noise = np.random.normal(0, noise_sigma, size=(3,))
                next_state[:3] += noise

                noisy_trajectory.append(next_state)
                current_state = next_state

            return np.array(noisy_trajectory)

        colors = ['red', 'blue', 'green']

        for idx, sigma in enumerate(noise_sigmas):
            noisy_trajectory = simulate_with_noise(trajectory, sigma)
            ax.plot3D(noisy_trajectory[:, 0], noisy_trajectory[:, 1], noisy_trajectory[:, 2], color=colors[idx], label=f'Noise {sigma}', linestyle='dashed')
            # 마지막 상태 포지션 출력
            last_position = noisy_trajectory[-1, :3]
            print(f'Last state with noise {sigma}: Position = {last_position}')
        
            # 충돌 여부 체크
            collision = False
            for sphere_center, sphere_radius in [([0.3, 0.3, 0.3], 0.1), ([0.5, 0.5, 0.6], 0.1), ([0.7, 0.7, 0.7], 0.1)]:
                distance = np.linalg.norm(last_position - np.array(sphere_center))
                if distance < sphere_radius:
                    collision = True
                    break
            if collision:
                print(f"Collision detected with noise {sigma}!")
            else:
                print(f"No collision detected with noise {sigma}.")
        ax.legend(prop = {'size':20})

        self._is_interrupted = False
        for i in range(self.T):
            self.quads['q1']['position'] = [trajectory[i, 0, 0], trajectory[i, 1, 0], trajectory[i, 2, 0]]
            self.quads['q1']['orientation'] = [trajectory[i, 6, 0], trajectory[i, 7, 0], trajectory[i, 8, 0]]
            self.update()
            if self._is_interrupted:
                return
        self._is_interrupted = False



    def play_with_bias(self, logger_folder=None, no_iter=-1, noise_sigmas=[0.0, 0.001, 0.001], biases=[0.0005, 0.0005, 0.001]):

        fig, ax = super().create_plot(figsize=(8, 8), xlim=(0, 1.1), ylim=(0, 1.1), zlim=(0, 1.1), is_3d=True, is_equal=False)

        def draw_sphere(xx, yy, zz, rr):
            u, v = np.mgrid[0:2 * np.pi:10j, 0:np.pi:10j]
            x = xx + rr * np.cos(u) * np.sin(v)
            y = yy + rr * np.sin(u) * np.sin(v)
            z = zz + rr * np.cos(v)
            ax.plot_wireframe(x, y, z, color="silver", alpha=0.6)

        draw_sphere(0.3, 0.3, 0.3, 0.1)
        draw_sphere(0.5, 0.5, 0.6, 0.1)
        draw_sphere(0.7, 0.7, 0.7, 0.1)
        self.quads = {'q1': {'position': [0, 0, 0], 'orientation': [0, 0, 0], 'L': 0.1}}
        for key in self.quads:
            self.quads[key]['l1'], = ax.plot([], [], [], color='deepskyblue', linewidth=3, antialiased=False)
            self.quads[key]['l2'], = ax.plot([], [], [], color='skyblue', linewidth=3, antialiased=False)
            self.quads[key]['hub'], = ax.plot([], [], [], marker='o', color='orange', markersize=10, antialiased=False)

        trajectory = np.asarray(logger.read_from_json(logger_folder, no_iter)["trajectory"])
        ax.plot3D(trajectory[:, 0, 0], trajectory[:, 1, 0], trajectory[:, 2, 0], label='No Noise', color='black')
        ax.legend('No noise')

        # no noise의 경우
        last_position = trajectory[-1, :3]
        print(f'Last state with no noise: Position = {last_position}')

        collision = False
        for sphere_center, sphere_radius in [([0.3, 0.3, 0.3], 0.1), ([0.5, 0.5, 0.6], 0.1), ([0.7, 0.7, 0.7], 0.1)]:
            distance = np.linalg.norm(last_position - np.array(sphere_center))
            if distance < sphere_radius:
                collision = True
                break
        if collision:
            print(f"Collision detected with no noise!")
        else:
            print(f"No collision detected with no noise.")

        def simulate_with_noise_and_bias(trajectory, noise_sigma, bias):
            noisy_trajectory = []
            current_state = trajectory[0, :12].flatten()  # 초기 상태

            for t in range(trajectory.shape[0]):
                output = trajectory[t, 12:16].flatten()  # 계획된 output
                next_state = self.evaluate_dynamics(current_state, output)  # 시스템의 다이나믹스 적용
                # Position에 bias 및 noise 추가
                noise = np.random.normal(0, noise_sigma, size=(3,))
                next_state[:3] += (noise + bias)

                noisy_trajectory.append(next_state)
                current_state = next_state

            return np.array(noisy_trajectory)

        colors = ['red', 'blue', 'green', 'purple']
        labels = [
        'Bias 0.0005',
        'Noise 0.001 + Bias 0.0005',
        'Noise 0.001 + Bias 0.002']

        for idx, (sigma, bias) in enumerate(zip(noise_sigmas + [0], biases)):
            noisy_trajectory = simulate_with_noise_and_bias(trajectory, sigma, bias)
            label = f'Noise {sigma}' if bias == 0 else f'Bias {bias}' if sigma == 0 else f'Noise {sigma} + Bias {bias}'
            ax.plot3D(noisy_trajectory[:, 0], noisy_trajectory[:, 1], noisy_trajectory[:, 2], color=colors[idx], label=label, linestyle='dashed')
        
            # 마지막 상태 포지션 출력
            last_position = noisy_trajectory[-1, :3]
            print(f'Last state with noise {sigma} and bias {bias}: Position = {last_position}')
        
            # 충돌 여부 체크
            collision = False
            for sphere_center, sphere_radius in [([0.3, 0.3, 0.3], 0.1), ([0.5, 0.5, 0.6], 0.1), ([0.7, 0.7, 0.7], 0.1)]:
                distance = np.linalg.norm(last_position - np.array(sphere_center))
                if distance < sphere_radius:
                    collision = True
                    break
            if collision:
                print(f"Collision detected with noise {sigma} and bias {bias}!")
            else:
                print(f"No collision detected with noise {sigma} and bias {bias}.")
        ax.legend(loc='upper left', prop = {'size':20})

        self._is_interrupted = False
        for i in range(self.T):
            self.quads['q1']['position'] = [trajectory[i, 0, 0], trajectory[i, 1, 0], trajectory[i, 2, 0]]
            self.quads['q1']['orientation'] = [trajectory[i, 6, 0], trajectory[i, 7, 0], trajectory[i, 8, 0]]
            self.update()
            if self._is_interrupted:
                return
        self._is_interrupted = False

    def plot_noisy_train(self, folders, noise_sigmas=None, biases=None):
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': '3d'})
        colors_map = {
            "QuadCopter_0": 'black',
            "QuadCopter_1": 'red',
            "QuadCopter_2": 'blue',
            "QuadCopter_3": 'green',
            "QuadCopter_5": 'orange',
            "QuadCopter_6": 'cyan',
            "QuadCopter_7": 'lightblue',
            "QuadCopter_8": 'pink',
            "QuadCopter_14": 'purple'
        }
        labels_map = {
            "QuadCopter_0": 'No Noise',
            "QuadCopter_1": 'Noise 0.001',
            "QuadCopter_2": 'Noise 0.005',
            "QuadCopter_3": 'Noise 0.002',
            "QuadCopter_5": 'Bias 0.0005',
            "QuadCopter_6": 'Bias 0.001',
            "QuadCopter_7": 'Bias 0.002',
            "QuadCopter_8": 'Bias 0.005',
            "QuadCopter_14": 'Noise 0.005_upgraded'
        }

        labels = [labels_map[folder] for folder in folders]
        colors = [colors_map[folder] for folder in folders]

        def draw_sphere(xx, yy, zz, rr):
            u, v = np.mgrid[0:2 * np.pi:10j, 0:np.pi:10j]
            x = xx + rr * np.cos(u) * np.sin(v)
            y = yy + rr * np.sin(u) * np.sin(v)
            z = zz + rr * np.cos(v)
            ax.plot_wireframe(x, y, z, color="silver", alpha=0.6)

        draw_sphere(0.3, 0.3, 0.3, 0.1)
        draw_sphere(0.5, 0.5, 0.6, 0.1)
        draw_sphere(0.7, 0.7, 0.7, 0.1)

        for idx, folder in enumerate(folders):
            trajectory = np.asarray(logger.read_from_json(folder, -1)["trajectory"])
            ax.plot3D(trajectory[:, 0, 0], trajectory[:, 1, 0], trajectory[:, 2, 0], color=colors[idx], label=labels[idx])

            last_position = trajectory[-1, :3]
            print(f'Last state for {labels[idx]}: Position = {last_position}')

            collision = False
            for sphere_center, sphere_radius in [([0.3, 0.3, 0.3], 0.1), ([0.5, 0.5, 0.6], 0.1), ([0.7, 0.7, 0.7], 0.1)]:
                distance = np.linalg.norm(last_position - np.array(sphere_center))
                if distance < sphere_radius:
                    collision = True
                    break
            if collision:
                print(f"Collision detected for {labels[idx]}!")
            else:
                print(f"No collision detected for {labels[idx]}.")

        ax.set_xlim(0, 1.1)
        ax.set_ylim(0, 1.1)
        ax.set_zlim(0, 1.1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(loc='upper left', prop = {'size':20})

    def plot_model(self, folders):
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': '3d'})
        colors_map = {
            "QuadCopter_model_based": 'red',
            "QuadCopter_0": 'blue'
        }
        labels_map = {
            "QuadCopter_model_based": 'Model-Based dynamics',
            "QuadCopter_0": 'Learned dynamics'
        }

        def draw_sphere(xx, yy, zz, rr):
            u, v = np.mgrid[0:2 * np.pi:10j, 0:np.pi:10j]
            x = xx + rr * np.cos(u) * np.sin(v)
            y = yy + rr * np.sin(u) * np.sin(v)
            z = zz + rr * np.cos(v)
            ax.plot_wireframe(x, y, z, color="silver", alpha=0.6)

        draw_sphere(0.3, 0.3, 0.3, 0.1)
        draw_sphere(0.5, 0.5, 0.6, 0.1)
        draw_sphere(0.7, 0.7, 0.7, 0.1)

        for idx, folder in enumerate(folders):
            trajectory = np.asarray(logger.read_from_json(folder, -1)["trajectory"])
            ax.plot3D(trajectory[:, 0, 0], trajectory[:, 1, 0], trajectory[:, 2, 0], color=colors_map[folder], label=labels_map[folder])

            last_position = trajectory[-1, :3]
            print(f'Last state for {labels_map[folder]}: Position = {last_position}')

            collision = False
            for sphere_center, sphere_radius in [([0.3, 0.3, 0.3], 0.1), ([0.5, 0.5, 0.6], 0.1), ([0.7, 0.7, 0.7], 0.1)]:
                distance = np.linalg.norm(last_position - np.array(sphere_center))
                if distance < sphere_radius:
                    collision = True
                    break
            if collision:
                print(f"Collision detected for {labels_map[folder]}!")
            else:
                print(f"No collision detected for {labels_map[folder]}.")
                

        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_zlim(0, 1.05)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(loc='upper left', prop = {'size':12})
        plt.show()