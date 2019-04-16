from __future__ import division
import numpy as np
import slam_utils
import tree_extraction
import math
from scipy.stats import chi2
from numpy.linalg import multi_dot



def motion_model(u, dt, ekf_state, vehicle_params):
    '''
    Computes the discretized motion model for the given vehicle as well as its Jacobian

    Returns:
        f(x,u), a 3x1 vector corresponding to motion x_{t+1} - x_t given the odometry u.

        df/dX, the 3x3 Jacobian of f with respect to the vehicle state (x, y, phi)
    '''
    ###
    # Implement the vehicle model and its Jacobian you derived.
    ###
    #u = (ve, alpha) 
    a = vehicle_params['a']
    b = vehicle_params['b']
    L = vehicle_params['L']
    H = vehicle_params['H']
    phi = ekf_state['x'][2]
    ve = u[0]
    alpha = u[1]    
    vc = ve/(1-math.tan(alpha)*H/L)
    
    motion = np.zeros(3)
    motion[0] = dt*(vc*math.cos(phi)-vc/L*math.tan(alpha)*(a*math.sin(phi)+b*math.cos(phi)))
    motion[1] = dt*(vc*math.sin(phi)+vc/L*math.tan(alpha)*(a*math.cos(phi)-b*math.sin(phi)))
    motion[2] = dt*(vc/L*math.tan(alpha))
    
    G = np.identity(3)
    G[0,2] = dt*(-vc*math.sin(phi)-vc/L*math.tan(alpha)*(a*math.cos(phi)-b*math.sin(phi)))
    G[1,2] = dt*(vc*math.cos(phi)-vc/L*math.tan(alpha)*(a*math.sin(phi)+b*math.cos(phi)))

    return motion, G

def odom_predict(u, dt, ekf_state, vehicle_params, sigmas):
    '''
    Perform the propagation step of the EKF filter given an odometry measurement u 
    and time step dt where u = (ve, alpha) as shown in the vehicle/motion model.

    Returns the new ekf_state.
    '''
    motion, G = motion_model(u, dt, ekf_state, vehicle_params)
    n = len(ekf_state['x'])
    ekf_state['x'][0:3] += motion
    ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
    
    Gt = np.identity(n)
    Gt[0:3,0:3] = G   
    R = np.zeros([n,n])
    R[0,0] = sigmas['xy']**2
    R[1,1] = sigmas['xy']**2
    R[2,2] = sigmas['phi']**2
    ekf_state['P'] = np.matmul(np.matmul(Gt,ekf_state['P']),Gt.T) + R
    ekf_state['P'] = slam_utils.make_symmetric(ekf_state['P'])
    #P = np.linalg.cholesky(ekf_state['P'])
    
    return ekf_state

def gps_update(gps, ekf_state, sigmas):
    '''
    Perform a measurement update of the EKF state given a GPS measurement (x,y).

    Returns the updated ekf_state.
    '''
    n = len(ekf_state['x'])
    H = np.zeros([2,n])
    H[:,0:3] = np.array([[1,0,0],[0,1,0]])
    Q = (sigmas['gps']**2)*np.identity(2)    
    r = gps - ekf_state['x'][0:2]
    S = H.dot(ekf_state['P']).dot(H.T)+Q.T
    d = (r.reshape([1,2])).dot(np.linalg.inv(S)).dot(r.reshape([2,1]))
    if d <= chi2.ppf(0.999, df=2):
        K = ekf_state['P'].dot(H.T).dot(np.linalg.inv(S))
        ekf_state['x'] += K.dot(r.reshape([2,1])).reshape(n)
        ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
        ekf_state['P'] = (np.identity(n) - K.dot(H)).dot(ekf_state['P'])
        ekf_state['P'] = slam_utils.make_symmetric(ekf_state['P'])
    ###
    # Implement the GPS update.
    ###
    
    return ekf_state

def laser_measurement_model(ekf_state, landmark_id):
    ''' 
    Returns the measurement model for a (range,bearing) sensor observing the
    mapped landmark with id 'landmark_id' along with its jacobian. 

    Returns:
        h(x, l_id): the 2x1 predicted measurement vector [r_hat, theta_hat].

        dh/dX: For a measurement state with m mapped landmarks, i.e. a state vector of
                dimension 3 + 2*m, this should return the full 2 by 3+2m Jacobian
                matrix corresponding to a measurement of the landmark_id'th feature.
    '''
    n = len(ekf_state['x'])
    xv = ekf_state['x'][0]
    yv = ekf_state['x'][1]
    phi = ekf_state['x'][2]
    xl = ekf_state['x'][3+2*landmark_id]
    yl = ekf_state['x'][3+2*landmark_id+1]
    zhat = np.zeros([2,1])
    zhat[0] = np.sqrt((xl-xv)**2 + (yl-yv)**2)
    zhat[1] = math.atan2(yl-yv,xl-xv)-phi
    zhat[1] = slam_utils.clamp_angle(zhat[1])
    
    F = np.zeros([5,n])
    F[:3,:3] = np.identity(3)
    F[3:5,3+2*landmark_id:3+2*landmark_id+2] = np.identity(2)
    q = (xl-xv)**2 + (yl-yv)**2
    J = 1/q * np.array([[-np.sqrt(q)*(xl-xv),-np.sqrt(q)*(yl-yv),0,np.sqrt(q)*(xl-xv),np.sqrt(q)*(yl-yv)],\
        [yl-yv,-(xl-xv),-q,-(yl-yv),xl-xv]])
    H = J.dot(F)
    
    ###
    # Implement the measurement model and its Jacobian you derived
    ###

    return zhat, H
#
def initialize_landmark(ekf_state, tree):
    '''
    Initialize a newly observed landmark in the filter state, increasing its
    dimension by 2.

    Returns the new ekf_state.
    '''
    n = np.size(ekf_state['x'])
    xv = ekf_state['x'][0]
    yv = ekf_state['x'][1]
    phi = ekf_state['x'][2]
    xl = xv + tree[0]*np.cos(phi+tree[1])
    yl = yv + tree[0]*np.sin(phi+tree[1])
    ekf_state['x'] = np.concatenate([ekf_state['x'],np.array([xl,yl])])
    P = 1000*np.identity(n+2)
    P[:n,:n] = ekf_state['P']
    ekf_state['P'] = P
    ekf_state['num_landmarks'] += 1
    
    ###
    # Implement this function.
    ###

    return ekf_state


def compute_data_association(ekf_state, measurements, sigmas, params):
    '''
    Computes measurement data association.

    Given a robot and map state and a set of (range,bearing) measurements,
    this function should compute a good data association, or a mapping from 
    measurements to landmarks.

    Returns an array 'assoc' such that:
        assoc[i] == j if measurement i is determined to be an observation of landmark j,
        assoc[i] == -1 if measurement i is determined to be a new, previously unseen landmark, or,
        assoc[i] == -2 if measurement i is too ambiguous to use and should be discarded.
    '''

    if ekf_state["num_landmarks"] == 0:
        # set association to init new landmarks for all measurements
        return [-1 for m in measurements]   
    ###
    # Implement this function.
    ###
    assoc = [-1 for m in measurements]  

    n = ekf_state['num_landmarks']
    m = len(measurements)
    
    M = np.zeros([m,n])
    Q = np.eye(2)
    Q[0,0] = sigmas['range']**2
    Q[1,1] = sigmas['bearing']**2
    for j in range(n):
        zhat, H = laser_measurement_model(ekf_state, j)            
        S = multi_dot([H, ekf_state['P'], H.T]) + Q
        for i,item in enumerate(measurements):
            r = np.array([item[0],item[1]]).reshape((2,1)) - zhat
            M[i,j] = multi_dot([r.T, np.linalg.inv(S), r])
        
    alpha = chi2.ppf(0.95, df=2)
    K = alpha*np.ones([m,m])
    Mk = np.hstack([M,K])
    
    result = slam_utils.solve_cost_matrix_heuristic(Mk)
    beta = chi2.ppf(0.99, df=2)
    for i,j in result:
        if j<n:
            assoc[i] = j
        else:
            assoc[i] = -2
            if min(M[i,:])>beta:
                assoc[i] = -1 

    return assoc


def laser_update(trees, assoc, ekf_state, sigmas, params):
    '''
    Perform a measurement update of the EKF state given a set of tree measurements.

    trees is a list of measurements, where each measurement is a tuple (range, bearing, diameter).

    assoc is the data association for the given set of trees, i.e. trees[i] is an observation of the
    ith landmark. If assoc[i] == -1, initialize a new landmark with the function initialize_landmark
    in the state for measurement i. If assoc[i] == -2, discard the measurement as 
    it is too ambiguous to use.

    The diameter component of the measurement can be discarded.

    Returns the ekf_state.
    '''
    for i,val in enumerate(assoc):
        if val == -1:
            ekf_state = initialize_landmark(ekf_state, trees[i])
            assoc[i] = ekf_state['num_landmarks']-1
            
    Q = np.identity(2)
    Q[0,0] = sigmas['range']**2
    Q[1,1] = sigmas['bearing']**2
    N = len(ekf_state['x'])
    for i,tree in enumerate(trees): 
        if assoc[i] != -2:
            zhat, H = laser_measurement_model(ekf_state, int(assoc[i]))
            K = ekf_state['P'].dot(H.T).dot(np.linalg.inv(H.dot(ekf_state['P']).dot(H.T)+Q.T))
            ekf_state['x'] += K.dot(tree[0:2] - zhat.reshape(2))
            ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
            ekf_state['P'] = (np.identity(N) - K.dot(H)).dot(ekf_state['P'])
            ekf_state['P'] = slam_utils.make_symmetric(ekf_state['P'])
    ###
    # Implement the EKF update for a set of range, bearing measurements.
    ###  
    return ekf_state


def run_ekf_slam(events, ekf_state_0, vehicle_params, filter_params, sigmas):
    last_odom_t = -1
    ekf_state = {
        'x': ekf_state_0['x'].copy(),
        'P': ekf_state_0['P'].copy(),
        'num_landmarks': ekf_state_0['num_landmarks']
    }
    
    state_history = {
        't': [0],
        'x': ekf_state['x'],
        'P': np.diag(ekf_state['P'])
    }

    if filter_params["do_plot"]:
        plot = slam_utils.init_plot()

    for i, event in enumerate(events):
        t = event[1][0]
        if i % 1000 == 0:
            print("t = {}".format(t))

        if event[0] == 'gps':
            gps_msmt = event[1][1:]
            ekf_state = gps_update(gps_msmt, ekf_state, sigmas)

        elif event[0] == 'odo':
            if last_odom_t < 0:
                last_odom_t = t
                continue
            u = event[1][1:]
            dt = t - last_odom_t
            ekf_state = odom_predict(u, dt, ekf_state, vehicle_params, sigmas)
            last_odom_t = t

        else:
            # Laser
            scan = event[1][1:]
            trees = tree_extraction.extract_trees(scan, filter_params)
            assoc = compute_data_association(ekf_state, trees, sigmas, filter_params)
            ekf_state = laser_update(trees, assoc, ekf_state, sigmas, filter_params)
            if filter_params["do_plot"]:
                slam_utils.do_plot(state_history['x'], ekf_state, trees, scan, assoc, plot, filter_params)

        
        state_history['x'] = np.vstack((state_history['x'], ekf_state['x'][0:3]))
        state_history['P'] = np.vstack((state_history['P'], np.diag(ekf_state['P'][:3,:3])))
        state_history['t'].append(t)

    return state_history


def main():
    odo = slam_utils.read_data_file("data/DRS.txt")
    gps = slam_utils.read_data_file("data/GPS.txt")
    laser = slam_utils.read_data_file("data/LASER.txt")

    # collect all events and sort by time
    events = [('gps', x) for x in gps]
    events.extend([('laser', x) for x in laser])
    events.extend([('odo', x) for x in odo])

    events = sorted(events, key = lambda event: event[1][0])

    vehicle_params = {
        "a": 3.78,
        "b": 0.50, 
        "L": 2.83,
        "H": 0.76
    }

    filter_params = {
        # measurement params
        "max_laser_range": 75, # meters

        # general...
        "do_plot": True,
        "plot_raw_laser": True,
        "plot_map_covariances": True

        # Add other parameters here if you need to...
    }

    # Noise values
    sigmas = {
        # Motion model noise
        "xy": 0.05,
        "phi": 0.5*np.pi/180,

        # Measurement noise
        "gps": 3,
        "range": 0.5,
        "bearing": 5*np.pi/180
    }

    # Initial filter state
    ekf_state = {
        "x": np.array( [gps[0,1], gps[0,2], 36*np.pi/180]),
        "P": np.diag([.1, .1, 1]),
        "num_landmarks": 0
    }

    run_ekf_slam(events, ekf_state, vehicle_params, filter_params, sigmas)

if __name__ == '__main__':
    main()
