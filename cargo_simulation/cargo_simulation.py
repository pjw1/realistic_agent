import carla
import random
from carla_painter import CarlaPainter
import numpy as np
import torch

from lanegcn_preprocess import get_preprocessed_data

import sys
sys.path.append('/home/jongwon/Desktop/realistic_vehicles/lanegcn')
from cargoverse_lanegcn import get_model
from utils import Logger, load_pretrain

sys.path.append('/home/jongwon/Desktop/realistic_vehicles/cargo_api')
from cargoverse.map_representation.cargoversemap_api import CargoverseMap

import pdb

def do_something(data):
    pass

def init_setting(num_vehicles, delta_sec, map_name = 'Town03'):
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    world = client.load_world(map_name)
    map = world.get_map()
    

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
    # set ego vehicle's role name to let CarlaViz know this vehicle is the ego vehicle
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
    
    return client, world, map, vehicle_ids

def save_tick(world, traj_dict, timestamp, city_name, vehicle_ids, near_threshold = 60):
                
    # get neighbor vehicle ids
    near_threshold = 60
    vehicle_locs = []
    for vehicle_id in vehicle_ids:
        loc = world.get_actor(vehicle_id).get_location()
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

def main():
    map_name = 'Town03'
    city_name = map_name[0] + map_name[-2:]
    num_vehicles = 200
    delta_sec = .1
    use_painter = True
    
    client, world, map, vehicle_ids = init_setting(num_vehicles, delta_sec, 
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
    
    # init for input dict
    curr_id = 0
    traj_dict_keys = ['TIMESTAMP', 'TRACK_ID', 'OBJECT_TYPE', 'X', 'Y', 'CITY_NAME']
    traj_dict = {}
    for key in traj_dict_keys:
        traj_dict[key] = []

    # tick to generate these actors in the game world
#     start_time = 2.
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
                              near_threshold = 60)
        
        # pass until saving 20 ticks 
        ts_list = sorted(list(set(traj_dict['TIMESTAMP'])))
        if len(ts_list) < 20:
            continue

        # run lanegcn
        
        assert len(ts_list) >= 20
        input_first_idx = traj_dict['TIMESTAMP'].index(ts_list[-20])
        
        input_dict = {}
        for key in traj_dict_keys:
            input_dict[key] = traj_dict[key][input_first_idx:]
        
        input_data = collate_fn([get_preprocessed_data(map_name, input_dict, cam, curr_id)])
        curr_id = curr_id + 1
        
        with torch.no_grad():
            output = net(input_data)
            results = [x.detach().cpu().numpy() for x in output["reg"]]  #(13, 6, 30, 2)
        
        if use_painter:
            pts = []
            z = ego_vehicle.get_location().z
            xy_np = np.array(results[0]).reshape(-1, 2)
            z_np = np.ones([xy_np.shape[0], 1]) * z
            xyz = np.hstack((xy_np, z_np)).tolist()
            painter.draw_points(xyz)
        # re init traj_dict



        # prune trajectories


if __name__ == "__main__":
    main()