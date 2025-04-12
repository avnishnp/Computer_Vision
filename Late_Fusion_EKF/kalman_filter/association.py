# ---------------------------------------------------------------------
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance


# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import kalman_filter.params as params 

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        
    def associate(self, track_list, meas_list, KF):
        
        # the following only works for at most one track and one measurement
        N = len(track_list) # N tracks
        M = len(meas_list) # M measurements
         
        
        self.association_matrix = np.matrix([]) # reset matrix
        self.unassigned_tracks = [] # reset lists
        self.unassigned_meas = []
        
        if N > 0: self.unassigned_tracks = list(range(N))
        if M > 0: self.unassigned_meas = list(range(M))
        if N > 0 and M > 0:
            self.association_matrix = np.inf * np.ones((N, M)) 
            for i in range(N): 
                track = track_list[i]
                for j in range(M):
                    meas = meas_list[j]
                    dist = self.MHD(track, meas, KF)
                    if self.gating(dist, sensor = meas.sensor):
                    #if True: ############# HERE!!!!!!!
                        self.association_matrix[i, j] = dist
                        #if i in self.unassigned_tracks: self.unassigned_tracks.remove(i)
                        #if j in self.unassigned_meas: self.unassigned_meas.remove(j)
        
                
    def get_closest_track_and_meas(self):

        # the following only works for at most one track and one measurement
        update_track = 0
        update_meas = 0
        
        # remove from list
        A = self.association_matrix
        if np.min(A) == np.inf:
            return np.nan, np.nan

        # get indices of minimum entry
        ij_min = np.unravel_index(np.argmin(A, axis=None), A.shape) 
        ind_track = ij_min[0]
        ind_meas = ij_min[1]

        # delete row and column for next update
        A = np.delete(A, ind_track, 0) 
        A = np.delete(A, ind_meas, 1)
        self.association_matrix = A

        # update this track with this measurement
        update_track = self.unassigned_tracks[ind_track] 
        update_meas = self.unassigned_meas[ind_meas]

        # remove this track and measurement from list
        self.unassigned_tracks.remove(update_track) 
        self.unassigned_meas.remove(update_meas)

        return update_track, update_meas   
 

    def gating(self, MHD, sensor): 

        p = params.gating_threshold #0.999 #params.gating_threshold #0.999 #0.95
        limit = chi2.ppf(params.gating_threshold, df = sensor.dim_meas)
        return MHD < limit
            
        
    def MHD(self, track, meas, KF):
        
        x = track.x
        H = meas.sensor.get_H(x) 
        #gamma = meas.z - H * x
        gamma = meas.z - meas.sensor.get_hx(x)
        S = H * track.P * H.transpose() + meas.R
        return gamma.transpose() * np.linalg.inv(S) * gamma
        
    
    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)