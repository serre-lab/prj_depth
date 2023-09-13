###### Stim Generator ######
# Celine Aubuchon


import bpy
import math
import os 
import numpy as np
import pickle
import sys
import random
from mathutils import Vector
from mathutils.noise import hetero_terrain as ht
#import polka_generator#import genPolkaTextureGrid

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)
import parameters_onego_random as parameters
# this re-loads the "parameters.py" Text every time
import importlib
importlib.reload(parameters)

### Blender Functions ###

################
def setupCamera(distance):
# Set the camera object to the correct distance and rotation
    
    # camera = bpy.data.objects['Camera']
    scene = bpy.context.scene
    
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    
    camera.location = (0, 0, distance)
    camera.rotation_euler = (0, 0, 0)
    
    scene.camera = camera  # defining the active cam for rendering
#    
#    cam = bpy.context.scene.camera.data
#    # set clipping distances
#    cam.clip_start = 0.01
#    cam.clip_end = 6
    
def setRenderProperties(res, samples):
## Set up the render properties using 'Cycles' engine with
 # 'samples' number of rendering samples
    
    # set engine to cycles
    bpy.context.scene.render.engine = 'CYCLES'
    
    # set to render using the GPU
    bpy.context.scene.cycles.device = 'GPU'
    
    # set Render sample count 
    bpy.context.scene.cycles.samples = samples
    
    # set square image dimensions
    bpy.context.scene.render.resolution_x = res
    bpy.context.scene.render.resolution_y = res
    
    # set the unit scale
    bpy.context.scene.unit_settings.scale_length = 1.0

    # set the unit system to 'METRIC'
    bpy.context.scene.unit_settings.system = 'METRIC'

    # 'METERS' is the base unit of the 'METRIC' system
    bpy.context.scene.unit_settings.length_unit = 'METERS'
    
def setupClothMaterial():
# set up the cloth material
    
    # remove previous cloth material
    for m in bpy.data.materials:
        if m.name == "Cloth":
            bpy.data.materials.remove(m)

    # Create a new material
    m = bpy.data.materials.new(name="Cloth")
    
    # Set up shader nodes and change defaultvalues
    m.use_nodes = True
    tex_image_node = m.node_tree.nodes.new("ShaderNodeTexImage")
    tex_coord_node = m.node_tree.nodes.new("ShaderNodeTexCoord")
    BSDF_node = m.node_tree.nodes.get("Principled BSDF")
    BSDF_node.inputs["Roughness"].default_value = 1
    
    # link the nodes together
    m.node_tree.links.new(tex_coord_node.outputs["UV"], tex_image_node.inputs[0])
    m.node_tree.links.new(tex_image_node.outputs["Color"], BSDF_node.inputs["Base Color"])
    
def setupSceneFloor():
## Create a 'floor' and set as collision object
    
    # add plane to the current collection (it will become the active object)
    bpy.ops.mesh.primitive_plane_add(size=parameters.terrain_size)
    
    # name the plane (active) as 'Floor'
    floor = bpy.context.object
    floor.name = 'Floor'
    
    # assign the floor (active) a collision modifier
    bpy.ops.object.modifier_add(type='COLLISION')

def setupRandTerrain(seed, size, height, noise_depth):
    
    # add random terrain based on parameters using A.N.T. Landscape plugin
    bpy.ops.mesh.landscape_add(mesh_size_x=size, 
        mesh_size_y=size, 
        random_seed=seed, 
        noise_type='hetero_terrain', 
        basis_type='PERLIN_ORIGINAL', 
        noise_depth=noise_depth, 
        height=height, 
        falloff_x=size, 
        falloff_y=size, 
        maximum=height, 
        minimum=-height, 
        refresh=True)
        
    # name the landscape (active) as 'Terrain'
    terrain = bpy.context.object
    terrain.name = 'Terrain'
    
    # assign the terrain (active) a collision modifier
    bpy.ops.object.modifier_add(type='COLLISION')


    

    
def setupLight(pos):

    # Create a new light datablock
    light_data = bpy.data.lights.new(name="New_Point_Light", type='POINT')

    # Change light parameters
    light_data.energy = 4000  # W power
    light_data.color = (1, 1, 1)  # RGB
    light_data.shadow_soft_size = 0.25  # radius
    light_data.cycles.max_bounces = 1024  # max bounces

    # Create a new object with the light datablock
    light_object = bpy.data.objects.new(name="New_Point_Light", object_data=light_data)

    # Link light object to the active collection of current view layer so that it'll appear in the current scene
    bpy.context.collection.objects.link(light_object)
    
    # Set light location
    light_object.location = pos

    # Make it active
    bpy.context.view_layer.objects.active = light_object
    light_object.select_set(True)
    
#    print(bb)
    
        
def setupCloth(subdivs, size, height, frame_end, use_image_textures):
# sets up and drops cloth (assumes there is a collision surface below the cloth)
    
    # add plane to the current collection (it will become the active object)
    bpy.ops.mesh.primitive_plane_add(size=size, location = (0, 0, height))
    
    # name the plane (active) as 'Cloth'
    cloth = bpy.context.object
    cloth.name = 'Cloth'
    
    # switch the edit mode and subdivide the cloth
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.subdivide(number_cuts=subdivs)
    
    # UV unwrap the cloth to account for subdivisions
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)
    
    # switch back to object mode
    bpy.ops.object.editmode_toggle()
    
    ## Set up cloth behavior
    # set the Cloth plane to act as a cloth
    bpy.ops.object.modifier_add(type='CLOTH')
    
    # change cloth settings from defaults
    bpy.context.object.modifiers["Cloth"].settings.quality = 10
    bpy.context.object.modifiers["Cloth"].settings.mass = 25
    bpy.context.object.modifiers["Cloth"].settings.tension_stiffness = 30
    bpy.context.object.modifiers["Cloth"].settings.compression_stiffness = 30
    bpy.context.object.modifiers["Cloth"].settings.shear_stiffness = 10
    bpy.context.object.modifiers["Cloth"].collision_settings.collision_quality = 5
    
    # set cache settings
    bpy.context.object.modifiers["Cloth"].point_cache.frame_end = frame_end
    
    # bake
    #bpy.ops.ptcache.bake(bake=True)
    bpy.ops.ptcache.bake_all()
    
    # set to last frame (drop the cloth)
    bpy.context.scene.frame_set(frame_end)
    
    # smooth shading
    bpy.ops.object.shade_smooth()
    
    # Apply the cloth modifier
    bpy.ops.object.modifier_apply(modifier="Cloth")
    
    # Add a subsurf modifier to smooth the cloth
    bpy.ops.object.modifier_add(type='SUBSURF')
    
    if use_image_textures:
        # Setup the cloth material
        setupClothMaterial()
    
        # Attach the material to the cloth
        cloth.data.materials.append(bpy.data.materials.get("Cloth"))
    else:
        # Attach the material to the cloth
        cloth.data.materials.append(bpy.data.materials.get("Musgrave"))

def setAmbient(val):
    background_node = bpy.data.worlds['World'].node_tree.nodes["Background"]
    background_node.inputs[0].default_value = val,val,val,1
    
  
def applyTexture(image_path = None):
# apply a texture image (at image_path) to the "Cloth" material
    m = bpy.data.materials.get("Cloth")
    
    tex = m.node_tree.nodes["Image Texture"]
    
    tex.image = bpy.data.images.load(image_path)
    

def setupCompositing(image_type): # image type refers to whether it is the stimulus or the mask of the stimulus
    # switch on nodes and get reference
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    
    # clear default nodes
    for node in tree.nodes:
        tree.nodes.remove(node)
        
    renderLayers_node = tree.nodes.new(type='CompositorNodeRLayers')
    renderLayers_node.location = -400,0
    
    composite_node = tree.nodes.new(type='CompositorNodeComposite')
    composite_node.location = 400,0
    
    alphaOver_node = tree.nodes.new(type='CompositorNodeAlphaOver')
    alphaOver_node.inputs[1].default_value = 0,0,0,1 # set background image as black
    alphaOver_node.location = -100,0
    
    tree.links.new(renderLayers_node.outputs["Image"], alphaOver_node.inputs[2])
    
    bpy.context.scene.view_settings.view_transform = 'Filmic'
    
    if image_type == "stimulus":
        tree.links.new(alphaOver_node.outputs["Image"], composite_node.inputs["Image"]) 

    if image_type == "depth_map":
        normalize_node = tree.nodes.new(type='CompositorNodeNormalize')
#        tree.links.new(renderLayers_node.outputs["Depth"], normalize_node.inputs["Value"])
#        tree.links.new(normalize_node.outputs["Value"], composite_node.inputs["Image"])
        tree.links.new(renderLayers_node.outputs["Depth"], composite_node.inputs["Image"])

        
    if image_type == "normal_map":
        tree.links.new(renderLayers_node.outputs["Normal"], composite_node.inputs["Image"])
        bpy.context.scene.view_settings.view_transform = 'Raw'
        
def saveImage(save_path, n_i, depth_bool=False):
# renders and saves image of current scene 

    if depth_bool:
        # names the image as a string of the seed, hdr, and texture information
        image_name = str(n_i) #+ '.tif'

        # sets save format to PNG
        scene = bpy.context.scene
        scene.render.image_settings.file_format='OPEN_EXR'
        scene.render.image_settings.color_mode = 'BW'
#        scene.render.image_settings.color_depth = '8'
    
        # sets filepath to save to
        scene.render.filepath = save_path + image_name
    
        # set up compositing for stimulus
        setupCompositing('depth_map')
    else:
        # names the image as a string of the seed, hdr, and texture information
        image_name = str(n_i) #+ '.png'

        # sets save format to PNG
        scene = bpy.context.scene
        scene.render.image_settings.file_format='PNG'
        scene.render.image_settings.color_mode = 'BW'
        scene.render.image_settings.color_depth = '8'
    
        # sets filepath to save to
        scene.render.filepath = save_path + image_name
    
        # set up compositing for stimulus
        setupCompositing('stimulus')
    
    # renders and saves image
    bpy.ops.render.render(write_still=1)
    
    print(image_name + " saved.")
    
    return save_path + image_name

def random_exclude_range(low1, high1, low2, high2):
    if np.random.rand() < 0.5:
        return np.random.uniform(low1, high1)
    else:
        return np.random.uniform(low2, high2)

def cleanup():

    # Select all objects
    bpy.ops.object.select_all(action='SELECT')
    
    # Delete all objects
    bpy.ops.object.delete()
    
    # Iterate over all materials and remove them
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
    
    # Iterate over all textures and remove them
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture)
        
    for item in bpy.data.meshes:
        bpy.data.meshes.remove(item)
    
    
    
########### SCRIPT #########################################################

# Settings

# generation settings
num_images = parameters.num_images

num_surfaces = parameters.num_surfaces
use_image_textures = parameters.use_image_textures
light_conds = parameters.light_levels
texture_types = parameters.num_tex_types

# stimulus settings
terrain_size = parameters.terrain_size
terrain_height = parameters.terrain_height
noise_depth = parameters.noise_depth

cloth_subdivs = parameters.cloth_subdivs
cloth_size = parameters.cloth_size
drop_height = parameters.drop_height
last_frame = parameters.last_frame

camera_distance = parameters.camera_distance
ambient_intensity_levels = parameters.ambient_intensity_levels

Texture_folder = bpy.path.abspath("//Textures_more_surfaces")
save_path =  bpy.path.abspath("//Rendered_Images_onego_random_noise_landscape_untitled//")
#Texture_folder = bpy.path.abspath("/cifs//data//tserre_lrs//projects//prj_depth//dataset//textures_folder//textures_imgs")
#save_path =  bpy.path.abspath("/cifs//data//tserre_lrs//projects//prj_depth//dataset//trial_ood//")

if not os.path.exists(f'{save_path}texture_tables'):
    os.mkdir(f'{save_path}texture_tables')


# Select the specific parameters you want to save
parameters_dict = {
    'num_images': parameters.num_images,
    'num_surfaces': parameters.num_surfaces,
    'use_image_textures': parameters.use_image_textures,
    'light_conds': parameters.light_levels,
    'texture_types': parameters.num_tex_types,
    'terrain_size': parameters.terrain_size,
    'terrain_height': parameters.terrain_height,
    'noise_depth': parameters.noise_depth,
    'cloth_subdivs': parameters.cloth_subdivs,
    'cloth_size': parameters.cloth_size,
    'drop_height': parameters.drop_height,
    'last_frame': parameters.last_frame,
    'camera_distance': parameters.camera_distance,
    'ambient_intensity_levels': parameters.ambient_intensity_levels
}

def save_parameters(parameters_dict, filename):
    with open(filename, 'wb') as f:
        pickle.dump(parameters_dict, f)

# Save the parameters to a file
save_parameters(parameters_dict, save_path + 'parameters.pkl')


# generate surface, save depth map, set up different lights and textures, render and save image
texture_index = 0
texture_table = []
for n_i in range(num_images):
    
    cleanup()

    # set up camera and render properties
    setupCamera(distance=camera_distance)
    setRenderProperties(samples=parameters.samples, res=parameters.res)
    setupSceneFloor()
    
    np.random.seed(n_i)
    
    ### generate surface
    rand_seed = np.random.randint(low = 0, high = num_surfaces)
    rand_noise_depth = np.random.uniform(low=noise_depth[0], high=noise_depth[1])
    setupRandTerrain(seed=rand_seed, 
        size=terrain_size, 
        height=terrain_height, 
        noise_depth=rand_noise_depth)
        
    setupCloth(subdivs=cloth_subdivs, 
        size=cloth_size, 
        height=drop_height, 
        frame_end=last_frame,
        use_image_textures=use_image_textures,)
#        seed = rand_seed)
        
    ### save depth map
    # sets filepath to save to
    saveImage(save_path + "depth_maps//", n_i, True)
    print("depth map " + str(n_i) + " saved.")
    
    rand_light = (random_exclude_range(parameters.light_posns_x[0], parameters.light_posns_x[1], parameters.light_posns_x[2], parameters.light_posns_x[3]), \
                  random_exclude_range(parameters.light_posns_y[0], parameters.light_posns_y[1], parameters.light_posns_y[2], parameters.light_posns_y[3]), \
                  random_exclude_range(parameters.light_posns_z[0], parameters.light_posns_z[1], parameters.light_posns_z[2], parameters.light_posns_z[3]), \
                                    )
    setupLight(rand_light)
  
    texture_img_path = Texture_folder + '//' + str(np.random.randint(low = 0, high = texture_types)) + '//texture_' + str(np.random.randint(low = 0, high = num_surfaces*light_conds)) + '.png'
    applyTexture(texture_img_path)
    
    setAmbient(ambient_intensity_levels)
    
    # render and save the image
    image_file = saveImage(save_path, n_i)
            
                
    texture_index = texture_index + 1
            
    cleanup() # delete current cloth and terrain
    
    # save data in a dictionary
    rowdict = {'n_i_and_seed': n_i, 'light_pos': rand_light,
               'noise_depth': rand_noise_depth,
               'depthmap_file': save_path + "depth_maps/" + str(n_i) + ".exr", 'image_file': save_path + str(n_i) + ".png", 
               'texture_file': texture_img_path}

    # add the dictionary to the list
    texture_table.append(rowdict)


    # if n_i is a multiple of 100, save the data
    if n_i % 5 == 0:
        with open(f'{save_path}texture_tables/texture_table_{n_i}.pkl', 'wb') as f:
            pickle.dump(texture_table, f)

        # remove previous file
        if n_i != 0:
            os.remove(f'{save_path}texture_tables/texture_table_{n_i - 5}.pkl')

# after the loop, save the entire list of dictionaries
with open(f'{save_path}texture_tables/texture_table.pkl', 'wb') as f:
    pickle.dump(texture_table, f)
        
            
            






        
            
            
