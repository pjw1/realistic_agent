import carla
import random
from carla_painter import CarlaPainter
import numpy as np
import torch
from lanegcn_preprocess import get_preprocessed_data
from cargoverse_lanegcn import get_model
from utils import Logger, load_pretrain
import sys
sys.path.append('/home/jongwon/Desktop/realistic_vehicles/cargo_api')
from cargoverse.map_representation.cargoversemap_api import CargoverseMap
from controller import PIDController
import argparse

import pdb

parser = argparse.ArgumentParser(description="Carla Realistic Vehicles Simulation")
parser.add_argument("--map", default="Town03", type=str, help="map name")
parser.add_argument("--num-vehicles", default=200, type=int, help="number of vehicles")
parser.add_argument("--start-tick", default=50, type=int, help="start point tick")
parser.add_argument("--delta-sec", default=0.1, type=float, help="inverse of fps")
parser.add_argument("--pred-delta", default=1.5, type=float, help="prediction interval")
parser.add_argument("--ctrl-delta", default=0.1, type=float, help="control interval")
parser.add_argument("--delta_sec", default=0.1, type=float, help="inverse of fps")
parser.add_argument("--near-threshold", default=50., type=float, help="interest radius from ego vehicle")
parser.add_argument("--use-pred", action="store_true", default=True)
parser.add_argument("--use-prune", action="store_true", default=True)
parser.add_argument("--use-pid", action="store_true", default=True)
parser.add_argument("--use-painter", action="store_true", default=False)

def do_something(data):
    pass

def init_setting(num_vehicles, delta_sec, map_name = 'Town03'):
    client = carla.Client('localhost', 2000)
    client.set_timeout(1000.0)
    
    world = client.load_world(map_name)
    
    map = world.get_map()
    
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    
    # set synchronous mode
    world.apply_settings(carla.WorldSettings(
        synchronous_mode=True,
        fixed_delta_seconds=delta_sec))

    # randomly spawn an ego vehicle and several other vehicles
    spawn_points = map.get_spawn_points()
    blueprints_vehicles = world.get_blueprint_library().filter("vehicle.*")

    ego_transform = spawn_points[random.randint(0, len(spawn_points) - 1)]
    other_vehicles_transforms = []
    for _ in range(num_vehicles):
        other_vehicles_transforms.append(spawn_points[random.randint(0, len(spawn_points) - 1)])

    blueprints_vehicles = [x for x in blueprints_vehicles if int(x.get_attribute('number_of_wheels')) == 4]
    
    # set ego vehicle
    #blueprints_vehicles[0] = world.get_blueprint_library().filter("vehicle.tesla.model3")
    blueprints_vehicles[0].set_attribute('role_name', 'ego') # or set to 'hero'
    batch = [carla.command.SpawnActor(blueprints_vehicles[0], ego_transform).then(carla.command.SetAutopilot(carla.command.FutureActor, True))]
    results = client.apply_batch_sync(batch, True)
    if not results[0].error:
        ego_vehicle = world.get_actor(results[0].actor_id)
    else:
        print('spawn ego error, exit')
        ego_vehicle = None
        return

    other_vehicles = []
    batch = []
    for i in range(num_vehicles):
        batch.append(carla.command.SpawnActor(blueprints_vehicles[i %(len(blueprints_vehicles)-1)+1], other_vehicles_transforms[i%len(other_vehicles_transforms)]).then(carla.command.SetAutopilot(carla.command.FutureActor, True)))

    # set autopilot for all these actors
    ego_vehicle.set_autopilot(True)
    results = client.apply_batch_sync(batch, True)
    for result in results:
        if not result.error:
            other_vehicles.append(result.actor_id)
    print(len(other_vehicles), ' vehicles have spawned')
    
    vehicle_ids = [ego_vehicle.id] + other_vehicles
    
    # attach a camera and a lidar to the ego vehicle
    camera = None
    blueprint_camera = world.get_blueprint_library().find('sensor.camera.rgb')
    blueprint_camera.set_attribute('image_size_x', '640')
    blueprint_camera.set_attribute('image_size_y', '480')
    blueprint_camera.set_attribute('fov', '110')
    blueprint_camera.set_attribute('sensor_tick', '0.1')
    transform_camera = carla.Transform(carla.Location(z=2.0))
    camera = world.spawn_actor(blueprint_camera, transform_camera, attach_to=ego_vehicle)
    camera.listen(lambda data: do_something(data))
    
    return client, world, map, tm, vehicle_ids

def save_tick(world, traj_dict, timestamp, city_name, vehicle_ids, near_threshold = 50):
    # get neighbor vehicle ids
#     near_threshold = 60
    vehicle_locs = []
    for vehicle_id in vehicle_ids:
        vehicle = world.get_actor(vehicle_id)
        loc = vehicle.get_location()
        vehicle_locs.append([loc.x, loc.y, loc.z])

    vlocs_np = np.array(vehicle_locs)

    vlocs_dist = np.sqrt(np.sum(np.square(vlocs_np[0:1, :2] - vlocs_np[:,:2]), axis=1))
    mask = vlocs_dist < near_threshold
    near_ids = np.where(mask)[0]
    num_near = sum(mask)
    near_locs = vlocs_np[mask, :2]

    # save tick to traj_dict
    tss = [timestamp for _ in range(num_near)]
    tids = np.array(vehicle_ids)[near_ids].tolist()
    ots = ['AV'] + ['OTHERS' for _ in range(num_near-1)]
    xs = near_locs[:,0].reshape(-1).tolist()
    ys = near_locs[:,1].reshape(-1).tolist()
    cns = [city_name for _ in range(num_near)]

    traj_dict['TIMESTAMP'] = traj_dict['TIMESTAMP'] + tss
    traj_dict['TRACK_ID'] = traj_dict['TRACK_ID'] + tids
    traj_dict['OBJECT_TYPE'] = traj_dict['OBJECT_TYPE'] + ots
    traj_dict['X'] = traj_dict['X'] + xs
    traj_dict['Y'] = traj_dict['Y'] + ys
    traj_dict['CITY_NAME'] = traj_dict['CITY_NAME'] + cns

    return traj_dict

def prune_preds(world, results, vids, cam, city_name, prune_threshold = 0.1):
    
    pred_prune_mask = np.zeros([results.shape[0], results.shape[1]])  # shape: (M, 6)
    for vehicle_idx, vehicle_id in enumerate(vids): #results.shape : (M, 6, 30, 2)
        # get valid lane 
        vehicle = world.get_actor(vehicle_id)
        vloc = vehicle.get_location()
        
        vori = vehicle.get_transform().get_forward_vector()
        
        
        adjacent_lane_ids = []
        dfs_lane_ids = []
        valid_lane_ids = []
        occupied_lane_ids = cam.get_lane_segments_containing_xy(vloc.x, vloc.y, city_name)

        for lane_id in occupied_lane_ids:
            adjacent_lane_ids = adjacent_lane_ids + cam.get_lane_segment_adjacent_ids(lane_id, city_name)
        adjacent_lane_ids = adjacent_lane_ids + occupied_lane_ids

        for lane_id in list(set(adjacent_lane_ids)):
            if lane_id is not None:
                try: # when lane length is 1, dfs prints error
                    dfs_lane_ids = dfs_lane_ids + cam.dfs(lane_id, city_name)
                except Exception as e:
#                         print(e)
                    pass

        for lane_ids in dfs_lane_ids:
            valid_lane_ids = valid_lane_ids + lane_ids
        
        
        kth_validity = []
        for k, kth_pred in enumerate(results[vehicle_idx]):
            check_idx = int(results.shape[2] * (1 - prune_threshold))
            
            wp = kth_pred[check_idx] # sample wp
            wp_occupied_lanes = cam.get_lane_segments_containing_xy(wp[0], wp[1], city_name)
            
            if len(wp_occupied_lanes) != 0:
                is_in_valid_lane = (wp_occupied_lanes[0] in valid_lane_ids)
            else:
                is_in_valid_lane = False
            
            if np.dot((kth_pred[-1] - kth_pred[0]), np.array([vori.x, vori.y])) > 0:
                is_heading_forward = True
            else:
                is_heading_forward = False
                
            kth_validity.append(is_in_valid_lane and is_heading_forward)
                
        for k, valid in enumerate(kth_validity):
            if valid:
                pred_prune_mask[vehicle_idx, k] = 1
    
    return pred_prune_mask

def compute_action(world, results, vids, vidx, k, tick_time, pred_delta):
    turn_control = PIDController(K_P=1.0)
    speed_control = PIDController(K_P=1.0)
    
    vehicle_id = vids[vidx]
    vehicle = world.get_actor(vehicle_id)

    # compute steer, throttle, ...
    ox = vehicle.get_transform().get_forward_vector().x
    oy = vehicle.get_transform().get_forward_vector().y
    rot = np.array([
        [ox, oy],
        [-oy, ox]])

    # vidx-th vehicle's target: 0-th traj, t-th wp
    t = int(tick_time % (pred_delta * 10))
    target = results[vidx, k, t]

    pos = vehicle.get_location()
    pos_np = np.array([pos.x, pos.y])

    target_direction = np.array([ox, oy]).dot(target - pos_np)
    diff = rot.dot(target - pos_np)

    speed = vehicle.get_velocity()
    speed = np.linalg.norm([speed.x, speed.y])

    u = np.array([diff[0], diff[1], 0.0])
    v = np.array([1.0, 0.0, 0.0])
    theta = np.arccos(np.dot(u, v) / np.linalg.norm(u))
    theta = theta if np.cross(u, v)[2] < 0 else -theta
#     if target_direction > 0:
    steer = turn_control.step(theta)

    # throttle
    average_through = int(pred_delta * 10)
    v = np.average(np.linalg.norm(results[vidx, k, 1:1+average_through] - results[vidx, k, :average_through], axis=1))
    target_speed = v * 10

    throttle = speed_control.step(target_speed - speed)
    
    return steer, throttle, target_direction

def compute_command(world, results, vids, vidx, k, tick_time, pred_delta):
    vehicle_id = vids[vidx]
    vehicle = world.get_actor(vehicle_id)

    # if all trajs are pruned -> set autopilot
    if np.unique(results[vidx, k]).shape[0] == 1: 
        command = [carla.command.SetAutopilot(vehicle_id, True)]
        return command

    steer, throttle, ori = compute_action(world, results, vids, vidx, k,
                                          tick_time, pred_delta)

    control = carla.VehicleControl()
    control.steer = np.clip(steer, -1., 1.)
    control.throttle = np.clip(throttle, 0.0, .5)
    control.brake = 0.0
    if ori < 0:
        control.throttle = 0.0
    control.manual_gear_shift = False

    command = [carla.command.ApplyVehicleControl(vehicle_id, control)]
    return command

def draw_preds(painter, results, show_k = -1):
    pts = []
    if show_k != -1:
        results = results[:, show_k]
    
    xy_np = np.array(results).reshape(-1, 2)
    z_np = np.ones([xy_np.shape[0], 1]) * 10
    xyz = np.hstack((xy_np, z_np)).tolist()
    painter.draw_points(xyz)

def main():

    
    args = parser.parse_args()
    
    map_name = args.map
    city_name = map_name[0] + map_name[-2:]
    num_vehicles = args.num_vehicles
    start_tick = args.start_tick
    
    delta_sec = args.delta_sec
    pred_delta = args.pred_delta
    ctrl_delta = args.ctrl_delta
    
    near_threshold = args.near_threshold
    
    use_pred = args.use_pred
    use_prune = args.use_prune
    use_pid = args.use_pid
    use_painter = args.use_painter
    
    client, world, map, tm, vehicle_ids = init_setting(num_vehicles, delta_sec, 
                                                   map_name=map_name)
    ego_vehicle = world.get_actor(vehicle_ids[0])

    cam = CargoverseMap()
    
    # initialize one painter
    if use_painter:
        painter = CarlaPainter('localhost', 8089)

    # load prediction module
    config, _, collate_fn, net, loss, post_process, opt = get_model()
    
    ckpt_path = '36.000.ckpt'
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    load_pretrain(net, ckpt["state_dict"])
    net.eval()
    
    
    tick = 0
    results = None
    
    # init for input dict
    curr_id = 0
    traj_dict_keys = ['TIMESTAMP', 'TRACK_ID', 'OBJECT_TYPE', 'X', 'Y', 'CITY_NAME']
    traj_dict = {}
    for key in traj_dict_keys:
        traj_dict[key] = []
    
    # tick to generate these actors in the game world
    anchor_ts = world.get_snapshot().timestamp.elapsed_seconds
    world.tick()
    
    while (True):
        world.tick()
        timestamp = world.get_snapshot().timestamp.elapsed_seconds

        # save tick to traj_dict
        traj_dict = save_tick(world, 
                              traj_dict, 
                              timestamp, 
                              city_name, 
                              vehicle_ids, 
                              near_threshold = near_threshold)
        
        # pass until saving 'start_tick'
        ts_list = sorted(list(set(traj_dict['TIMESTAMP'])))
        tick_time = len(ts_list)
        # To Do: backup log when dict gets too big
        # traj_dict = backup_log(...)
        
        # Prediction Module
        if use_pred and tick_time >= start_tick and tick_time % (pred_delta * 10) == 0:
            # pred trajs 
            input_first_idx = traj_dict['TIMESTAMP'].index(ts_list[-20])
            input_dict = {}
            for key in traj_dict_keys:
                input_dict[key] = traj_dict[key][input_first_idx:]
            
            preprocessed_data, vids=get_preprocessed_data(map_name,input_dict,cam,curr_id, painter)
            input_data = collate_fn([preprocessed_data])
            curr_id = curr_id + 1
            
            with torch.no_grad():
#                 pdb.set_trace()
                output = net(input_data)
                results = [x.detach().cpu().numpy() for x in output["reg"]][0]  

            # prune trajs
            if use_prune:
                pred_prune_mask = prune_preds(world, results, vids, cam, city_name, prune_threshold = 0.1)
                ms = pred_prune_mask.shape
                results = results * pred_prune_mask.reshape(ms[0], ms[1], 1, 1)
                
            if use_painter:
                draw_preds(painter, results, show_k=0)
                
        # Control Module
        if use_pid and results is not None and tick_time % (ctrl_delta * 10) == 0:
            command_batch = []
            
            #compute all near vehicle's control
            for vidx in range(results.shape[0]):
                 # find first not pruned traj of vidx's
                for k in range(results.shape[1]):
                    if np.unique(results[vidx, k]).shape[0] != 1:
                        break
                
                command = compute_command(world, results, vids, vidx, k, tick_time, pred_delta)
                command_batch = command_batch + command
                
            results_ = client.apply_batch_sync(command_batch, False)
                
if __name__ == "__main__":
    main()
