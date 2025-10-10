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
from pascal_voc_writer import Writer


# --- Geometric Transformations (Prerequisites) ---

def build_projection_matrix(w, h, fov):
    """
    Constructs the camera projection matrix K.
    """
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    """
    Calculates 2D projection of 3D coordinate (world coordinates) into the image plane.
    """
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2]

# --- Main Script Start ---
actor_list = []
try:
    # 1. Set up the simulator world
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    
    # Create output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving images and XML files to '{output_dir}/'")

    # Spawn vehicle (ego-vehicle)
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

    # 2. Get the camera params and compute the matrix
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()
    K = build_projection_matrix(image_w, image_h, fov)

    # 3. Spawn 50 NPC Vehicles
    for i in range(50):
        vehicle_bp = random.choice(bp_lib.filter('vehicle'))
        npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        if npc:
            npc.set_autopilot(True)
            actor_list.append(npc)

    # Initial tick and setup for OpenCV window
    world.tick()
    image = image_queue.get(True, 1.0)
    cv2.namedWindow('2D Bounding Box Preview (ESC or Q to exit)', cv2.WINDOW_AUTOSIZE)
    print("Starting data export and preview loop. Press 'ESC' or 'q' in the OpenCV window to stop.")


    # 4. EXPORT AND PREVIEW LOOP
    while True:
        # Retrieve the image
        world.tick()
        image = image_queue.get(True, 1.0)

        # FIX: Ensure a contiguous 3-channel array for OpenCV
        array = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        # Select the BGR channels (dropping A/alpha) and explicitly create a contiguous copy
        img = array[:, :, :3].copy() 

        # Get the camera matrix (inverse world transform)
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
        
        # Get ego vehicle transform for checks
        ego_transform = vehicle.get_transform()

        # Define output paths
        frame_path = os.path.join(output_dir, '%06d' % image.frame)

        # Save the image
        image.save_to_disk(frame_path + '.png')

        # Initialize the exporter (Writer expects image width/height as integers)
        writer = Writer(frame_path + '.png', image_w, image_h)

        for npc in world.get_actors().filter('*vehicle*'):
            if npc.id != vehicle.id:
                bb = npc.bounding_box
                dist = npc.get_transform().location.distance(ego_transform.location)
                
                if dist < 50:
                    forward_vec = ego_transform.get_forward_vector()
                    ray = npc.get_transform().location - ego_transform.location

                    # Check if the vehicle is in front
                    if forward_vec.dot(ray) > 0:
                        
                        # Get 8 vertices of the 3D bounding box
                        verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                        
                        # Initialize min/max projected coordinates
                        x_max = -10000.0
                        x_min = 10000.0
                        y_max = -10000.0
                        y_min = 10000.0

                        # Find the extremities of the 2D projection
                        for vert in verts:
                            p = get_image_point(vert, K, world_2_camera)
                            
                            if p[0] > x_max:
                                x_max = p[0]
                            if p[0] < x_min:
                                x_min = p[0]
                            if p[1] > y_max:
                                y_max = p[1]
                            if p[1] < y_min:
                                y_min = p[1]

                        # Clip values to image bounds and ensure integers for PASCAL VOC
                        x_min_int = int(max(1, x_min))
                        y_min_int = int(max(1, y_min))
                        x_max_int = int(min(image_w - 1, x_max))
                        y_max_int = int(min(image_h - 1, y_max))

                        # Check if the calculated bounding box is valid and inside bounds
                        if x_max_int > x_min_int and y_max_int > y_min_int:
                            
                            # Add the object to the XML writer (PASCAL VOC)
                            writer.addObject('vehicle', x_min_int, y_min_int, x_max_int, y_max_int)
                            
                            # Draw the 2D box on the OpenCV image (RED box)
                            cv2.rectangle(img, 
                                          (x_min_int, y_min_int), 
                                          (x_max_int, y_max_int), 
                                          (0, 0, 255), # BGR: Red
                                          1)
                            
        # Show the image with drawn bounding boxes
        cv2.imshow('2D Bounding Box Preview (ESC or Q to exit)', img)
        
        # Save the bounding boxes in the scene to the XML file
        writer.save(frame_path + '.xml')
        
        # Break the loop if the user presses 'q' or 'ESC'
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27: # 27 is the ASCII code for ESC
            break
        

finally:
    # Cleanup: Destroy all spawned actors and reset world settings
    print('\nScript interrupted. Destroying actors and resetting world settings.')
    client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    
    cv2.destroyAllWindows()
    print("Script finished and cleanup performed.")