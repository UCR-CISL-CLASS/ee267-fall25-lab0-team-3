# Code to find the Carla module from a running instance
import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# Import carla and other modules
import carla
import math
import random
import time
import queue
import numpy as np
import cv2

# --- Geometric Transformations (Required Functions) ---

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    """Constructs the camera projection matrix K."""
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    """Calculates 2D projection of 3D coordinate into the image plane."""
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    # (x, y ,z) -> (y, -z, x)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2]

def point_in_canvas(pos, img_h, img_w):
    """Return true if point is in canvas bounds"""
    if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
        return True
    return False

# --- Main Script Start ---
actor_list = []
try:
    # 1. Set up the simulator world
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    # Spawn main vehicle (ego vehicle)
    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    vehicle.set_autopilot(True)
    actor_list.append(vehicle)

    # Spawn camera sensor
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_init_trans = carla.Transform(carla.Location(z=2))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
    actor_list.append(camera)

    # Set up synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True 
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Create a queue to store and retrieve the sensor data
    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    # 2. Get the camera params and compute the matrices
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()

    # Calculate the projection matrices (K_b is for robustness checks)
    K = build_projection_matrix(image_w, image_h, fov)
    K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

    # Define the list of edge pairs for a 3D bounding box (8 vertices -> 12 edges)
    edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

    # 3. Spawn 50 NPC Vehicles
    for i in range(50):
        vehicle_bp = random.choice(bp_lib.filter('vehicle'))
        npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        if npc:
            npc.set_autopilot(True)
            actor_list.append(npc) # Track NPCs for cleanup

    # 4. Get Level Bounding Boxes (Traffic Lights and Signs)
    bounding_box_set_level = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
    bounding_box_set_level.extend(world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))

    # Initial tick to retrieve the first image and set up OpenCV
    world.tick()
    image = image_queue.get(True, 1.0) 
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    cv2.namedWindow('ImageWindowName', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('ImageWindowName', img)
    cv2.waitKey(1)

    # 5. RENDERING GAME LOOP

    print("Starting rendering loop (Red boxes for level, Blue for vehicles). Press 'q' to exit.")

    while True:
        # Retrieve and reshape the image
        world.tick()
        image = image_queue.get(True, 1.0)
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

        # Update the world-to-camera matrix 
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
        
        # Get ego vehicle transform for checks
        ego_transform = vehicle.get_transform()
        forward_vec = ego_transform.get_forward_vector()
        
        # --- Draw Level Bounding Boxes (Traffic Lights/Signs) in RED ---
        for bb in bounding_box_set_level:
            # Filter for distance (< 50m) and being in front (> 0)
            if bb.location.distance(ego_transform.location) < 50:
                ray = bb.location - ego_transform.location
                if forward_vec.dot(ray) > 0:
                    verts = [v for v in bb.get_world_vertices(carla.Transform())]
                    
                    for edge in edges:
                        p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                        p2 = get_image_point(verts[edge[1]], K, world_2_camera)

                        # Draw the edge in RED (BGR: 0, 0, 255)
                        cv2.line(img, 
                                 (int(p1[0]), int(p1[1])), 
                                 (int(p2[0]), int(p2[1])), 
                                 (0, 0, 255, 255), 
                                 1)

        # --- Draw Vehicle Bounding Boxes in BLUE ---
        for npc in world.get_actors().filter('*vehicle*'):

            if npc.id != vehicle.id:
                bb = npc.bounding_box
                dist = npc.get_transform().location.distance(ego_transform.location)

                if dist < 50:
                    ray = npc.get_transform().location - ego_transform.location

                    if forward_vec.dot(ray) > 0:
                        verts = [v for v in bb.get_world_vertices(npc.get_transform())]

                        for edge in edges:
                            p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                            p2 = get_image_point(verts[edge[1]], K, world_2_camera)

                            p1_in_canvas = point_in_canvas(p1, image_h, image_w)
                            p2_in_canvas = point_in_canvas(p2, image_h, image_w)

                            if not p1_in_canvas and not p2_in_canvas:
                                continue

                            # Robustness: Handle vertices behind the camera
                            ray0 = verts[edge[0]] - camera.get_transform().location
                            ray1 = verts[edge[1]] - camera.get_transform().location
                            cam_forward_vec = camera.get_transform().get_forward_vector()

                            if not (cam_forward_vec.dot(ray0) > 0):
                                p1 = get_image_point(verts[edge[0]], K_b, world_2_camera)
                            
                            if not (cam_forward_vec.dot(ray1) > 0):
                                p2 = get_image_point(verts[edge[1]], K_b, world_2_camera)

                            # Draw the edge in BLUE (BGR: 255, 0, 0)
                            cv2.line(img, 
                                     (int(p1[0]),int(p1[1])), 
                                     (int(p2[0]),int(p2[1])), 
                                     (255, 0, 0, 255), 
                                     1)        

        cv2.imshow('ImageWindowName', img)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    # Cleanup: Destroy all spawned actors
    print('Destroying actors and resetting world settings.')
    client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

    # Reset world settings
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    
    cv2.destroyAllWindows()
    print("Script finished and cleanup performed.")