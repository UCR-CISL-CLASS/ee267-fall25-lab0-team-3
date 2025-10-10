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

# --- Geometric Transformations (Prerequisites) ---

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    """
    Constructs the camera projection matrix K.
    :param w: Image width
    :param h: Image height
    :param fov: Field of View
    :param is_behind_camera: Adjust focal length if projecting from behind the camera
    :return: 3x3 projection matrix K
    """
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
    """
    Calculates 2D projection of 3D coordinate (world coordinates) into the image plane.
    :param loc: carla.Location object in world coordinates
    :param K: Camera projection matrix
    :param w2c: World-to-Camera inverse matrix
    :return: 2D image coordinates [x, y]
    """
    # 1. Format the input coordinate (loc is a carla.Location object)
    point = np.array([loc.x, loc.y, loc.z, 1])

    # 2. Transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # 3. Change from UE4's coordinate system to a standard (y, -z, x) system
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # 4. Project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)

    # 5. Normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]

# --- Main Script Start ---
try:
    # 1. Set up the simulator world

    client = carla.Client('localhost', 2000)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    # Spawn vehicle (ego-vehicle)
    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    vehicle.set_autopilot(True)

    # Spawn camera sensor
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_init_trans = carla.Transform(carla.Location(z=2))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

    # Set up synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Create a queue to store and retrieve the sensor data
    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    # 2. Get the camera params and compute the matrices

    # Get the world to camera matrix (inverse camera transform)
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    # Get the attributes from the camera blueprint
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()

    # Calculate the camera projection matrix to project from 3D -> 2D
    K = build_projection_matrix(image_w, image_h, fov)

    # 3. setup the boudning boxes (Prerequisites)

    # Set up the set of bounding boxes from the level (Traffic Lights and Signs)
    bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
    bounding_box_set.extend(world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))

    # Define the list of edge pairs for a 3D bounding box (8 vertices -> 12 edges)
    edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

    # Initial tick to retrieve the first image and set up OpenCV
    world.tick()
    image = image_queue.get(True, 1.0) # wait up to 1 second for the first image
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    cv2.namedWindow('ImageWindowName', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('ImageWindowName', img)
    cv2.waitKey(1)

    # 4. RENDERING THE BOUNDING BOXES (The Game Loop)

    print("Starting rendering loop. Press 'q' in the OpenCV window to exit.")

    while True:
        # Retrieve and reshape the image
        world.tick()
        image = image_queue.get(True, 1.0)
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

        # Update the world-to-camera matrix (in case the camera moved)
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        for bb in bounding_box_set:
            # Filter for distance from ego vehicle (< 50m)
            if bb.location.distance(vehicle.get_transform().location) < 50:
                
                # Check if the bounding box is IN FRONT OF THE CAMERA (using dot product)
                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = bb.location - vehicle.get_transform().location
                
                if forward_vec.dot(ray) > 0:
                    # Get the 8 vertices in world coordinates
                    verts = [v for v in bb.get_world_vertices(carla.Transform())]
                    
                    for edge in edges:
                        # Project the two vertices of the edge into image coordinates
                        p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                        p2 = get_image_point(verts[edge[1]], K, world_2_camera)

                        # Draw the edge into the camera output (Red line for level objects)
                        cv2.line(img, 
                                 (int(p1[0]), int(p1[1])), 
                                 (int(p2[0]), int(p2[1])), 
                                 (0, 0, 255, 255), # BGR color: Red
                                 1)

        # Draw the image into the OpenCV display window
        cv2.imshow('ImageWindowName', img)

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    # Cleanup: Stop camera, destroy actors, reset settings, and close windows
    if 'camera' in locals() and camera.is_alive:
        camera.stop()
        camera.destroy()
    if 'vehicle' in locals() and vehicle.is_alive:
        vehicle.destroy()

    # Reset world settings
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    
    # Close the OpenCV display window when the game loop stops
    cv2.destroyAllWindows()
    print("Script finished and cleanup performed.")