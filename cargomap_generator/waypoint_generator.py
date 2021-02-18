import carla
import matplotlib.pyplot as plt
import numpy as np
import math
import pdb
import os
from carla_painter import CarlaPainter
import random
import pickle
import pandas as pd

def do_something(data):
    pass

def wp_loc_(wp):
    return [wp.transform.location.x, wp.transform.location.y, wp.transform.location.z]

def main():
    
    wp_distance = .5
    xodr_root = 'xodr_files/'
    wp_root = 'wp_dicts/'
    
#     painter = CarlaPainter('localhost', 8089)

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    print("client has connected")
    
    available_maps = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05']
    
    for map_name in available_maps:
        world = client.load_world(map_name)
        print(map_name+' has loaded')

        map_ = world.get_map()

        # Save xodr file
        xodr_fname = xodr_root + map_name + '.xodr'
        map_.save_to_disk(xodr_fname)

        ### Generate waypoint_dict.csv 
        waypoint_list = map_.generate_waypoints(wp_distance)
        wp_loc = [wp_loc_(wp) for wp in waypoint_list]
        wp_dict = {'road_id':[], 'lane_id':[], 'loc':[], 'is_junction':[]} 
        wp_dict_type = {}

        for i, wp in enumerate(waypoint_list):

            wp_dict['road_id'].append(wp.road_id)
            wp_dict['lane_id'].append(wp.lane_id)
            wp_dict['loc'].append(wp_loc_(wp))
            wp_dict['is_junction'].append(wp.is_junction)

        wp_df = pd.DataFrame(wp_dict)
        
        csv_fname = wp_root + 'wp_dict_' + map_name + '.csv'
        wp_df.to_csv(csv_fname)
    
    ###
    
    pdb.set_trace()
    
    if lidar is not None:
        lidar.stop()
        lidar.destroy()
    if camera is not None:
        camera.stop()
        camera.destroy()
    if ego_vehicle is not None:
        ego_vehicle.destroy()
    if other_vehicles is not None:
        client.apply_batch([carla.command.DestroyActor(x) for x in other_vehicles])   

if __name__ == '__main__':
    main()