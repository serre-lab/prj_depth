###### Stim Generator ######
# Celine Aubuchon


import bpy
import math
import os 
import numpy as np
#import pip
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
    bpy.ops.mesh.primitive_plane_add(size=5)
    
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


    
def generateRandomSpheres(num_spheres, max_radius, max_height):
    """Generate random spheres as terrain"""

    # create a new collection for the spheres
    sphere_collection = bpy.data.collections.new('Spheres')
    bpy.context.scene.collection.children.link(sphere_collection)

    # create several spheres with random locations and sizes
    for i in range(num_spheres):
        # random location and size
        location = (random.uniform(-max_height, max_height), 
                    random.uniform(-max_height, max_height), 
#                    random.uniform(0, max_height)
                    0)
        size = random.uniform(0, max_radius)
        
        # add the sphere
        bpy.ops.mesh.primitive_uv_sphere_add(
            location=location,
            radius=size
        )

        # link the sphere to the new collection
        sphere_collection.objects.link(bpy.context.object)
        bpy.context.collection.objects.unlink(bpy.context.object)

    # add a large flat plane beneath the spheres
    bpy.ops.mesh.primitive_plane_add(size=4*max_height)
    sphere_collection.objects.link(bpy.context.object)
    bpy.context.collection.objects.unlink(bpy.context.object)

    # join all the objects (spheres and plane) into a single mesh
    bpy.ops.object.select_all(action='DESELECT')
    for obj in sphere_collection.objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.join()

    # rename the joined object
    obj.name = 'Terrain'
    
    ###########
    # smooth shading
    bpy.ops.object.shade_smooth()
    # Add a subsurf modifier to smooth the cloth
    bpy.ops.object.modifier_add(type='SUBSURF')
    
    bpy.ops.object.modifier_add(type='COLLISION')
    
def create_landscape(size, subdivs, seed = None):
#    context = bpy.context

    H = 0.4
    lacunarity = 1
    octaves = 1
    offset = 0.3
    noise_basis = 'PERLIN_ORIGINAL'
#    
#    random.seed(seed)
#    offset_x = random.uniform(-0, 0)  # Change range as needed
#    offset_y = random.uniform(-0, 0)  # Change range as needed

#    bpy.ops.mesh.primitive_grid_add(
#        size=size,
#        x_subdivisions=subdivs,
#        y_subdivisions=subdivs
#        )
#    ob = context.object
#    # rename the joined object
#    ob.name = 'Terrain'
#    
#    me = ob.data
#    sk = ob.shape_key_add(name="Basis")

#    sk = ob.shape_key_add(name=noise_basis)

#    for v in me.vertices:
#        sk.data[v.index].co.z = ht(v.co + Vector((offset_x, offset_y, 0)),
#                H,
#                lacunarity,
#                octaves,
#                offset,
#                noise_basis=noise_basis
#                )

    bpy.ops.mesh.primitive_grid_add(size=size, x_subdivisions=subdivs, y_subdivisions=subdivs)
    ob = bpy.context.object
    me = ob.data

    # Add a shape key to store the original state
#    bpy.ops.object.shape_key_add(from_mix=False)
    
    # Add another shape key for the Perlin noise
    sk = ob.shape_key_add(name="PerlinNoise")
    
    for v in me.vertices:
        co = Vector(v.co)
        z = ht(co + Vector((seed*5, seed*5, 0)), H, lacunarity, octaves, offset, noise_basis='PERLIN_ORIGINAL')
        # Scale the z values
        z = z * 0.75
        sk.data[v.index].co = Vector((co.x, co.y, z))

    ###########
    # smooth shading
    bpy.ops.object.shade_smooth()
    # Add a subsurf modifier to smooth the cloth
    bpy.ops.object.modifier_add(type='SUBSURF')
    
    bpy.ops.object.modifier_add(type='COLLISION')

    
def setupLight(pos):
#    light = bpy.data.objects.get('Light')#.select_set(True)
#    light.location = pos

    # Create a new point light if one doesn't exist
    bpy.ops.object.light_add(type='POINT', location=(0, 0, 0))
    light = bpy.context.object
    light.name = 'Light'

    light.data.use_nodes = True
    light.data.node_tree.nodes['Emission'].inputs['Strength'].default_value = 30
    light.data.node_tree.nodes['Emission'].inputs[0].default_value = 0.9,0.9,0.9,1
    
    light.location = pos
    
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
    
#    img_array_rgba = genPolkaTextureGrid(return_image = True)
#    
#    # Rearrange the image array to RGBA
#    img_array_rgba = np.zeros((im_np.shape[0], im_np.shape[1], 4))
#    img_array_rgba[..., 0:3] = im_np[..., 0:3] / 255.0  # Convert to 0-1 range
#    img_array_rgba[..., 3] = 1.0  # Alpha channel

#    # Flatten the array and convert it to a list
#    height, width = img_array_rgba.shape[0], img_array_rgba.shape[1]
#    pixels = list(img_array_rgba.flatten())

#    # Your flat array of RGBA float32s
#    assert len(pixels) == 4 * width * height

#    # Create image
#    # Note: choose if the image should have alpha here, but even if
#    # it doesn't, the array still needs to be RGBA
#    img = bpy.data.images.new('texture_img', width, height, alpha=False)

#    # Fast way to set pixels (since 2.83)
#    img.pixels.foreach_set(pixels)

#    # Pack the image into .blend so it gets saved with it
#    img.pack()
#    
#    tex.image = img
    

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
        tree.links.new(renderLayers_node.outputs["Depth"], normalize_node.inputs["Value"])
        tree.links.new(normalize_node.outputs["Value"], composite_node.inputs["Image"])
        
    if image_type == "normal_map":
        tree.links.new(renderLayers_node.outputs["Normal"], composite_node.inputs["Image"])
        bpy.context.scene.view_settings.view_transform = 'Raw'
        
def saveImage(save_path, n_i):
# renders and saves image of current scene 

#    # names the image as a string of the seed, hdr, and texture information
#    image_name = 'surf_' + str(seed) + '_type_' + str(type)+ \
#        '_texture_' + str(texture) + '_light_'+ str(light)+ '.tif'

    # names the image as a string of the seed, hdr, and texture information
    image_name = str(n_i)+ '.png'

    # sets save format to PNG
    scene = bpy.context.scene
    scene.render.image_settings.file_format='PNG'
    
    # sets filepath to save to
    scene.render.filepath = save_path + image_name
    
    # set up compositing for stimulus
    setupCompositing('stimulus')
    
    # renders and saves image
    bpy.ops.render.render(write_still=1)
    
    print(image_name + " saved.")
    
    return save_path + image_name + '.png'
        

def cleanup():
# delete the previous surface objects to get ready for the next
    bpy.data.objects['Cloth'].select_set(True)
    bpy.ops.object.delete()
#    try:
    bpy.data.objects['Terrain'].select_set(True)
    bpy.ops.object.delete()
#    except:
#        print('No Terrain')

def cleanup():
#    # iterate over all objects
#    for obj in bpy.data.objects:
#        # if the object is not a camera or light
#        if obj.type not in {'CAMERA', 'LIGHT'}:
#            # select the object
#            obj.select_set(True)
#            # delete the selected object
#            bpy.ops.object.delete()

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
use_image_textures = True
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

# Texture_folder = bpy.path.abspath("/media//data_cifs//projects//prj_depth//dataset//textures_folder//textures_imgs")
# save_path =  bpy.path.abspath("/media//data_cifs//projects//prj_depth//dataset//trial//")
Texture_folder = bpy.path.abspath("/cifs//data//tserre_lrs//projects//prj_depth//dataset//textures_folder//textures_imgs")
save_path =  bpy.path.abspath("/cifs//data//tserre_lrs//projects//prj_depth//dataset//trial//")

# ## create texture filepath dictionary
# textures = {}
# keys = np.arange(parameters.num_tex_types)

# for i in keys:
#     textures[i] = [os.path.join(Texture_folder + '//'+str(i)+'//', f) for f in os.listdir(Texture_folder + '//'+str(i)+'//')]


# setRenderProperties(samples=parameters.samples, res=parameters.res)
#setupSceneFloor()

#columns = ['surf', 'light_type', 'tex_type', 'depthmap_file', 'image_file', 'texture_file']
#data_table = pd.DataFrame(columns=columns)

# generate surface, save depth map, set up different lights and textures, render and save image
texture_index = 0
for n_i in range(num_images):
    
    cleanup()

    setRenderProperties(samples=parameters.samples, res=parameters.res)
    
    np.random.seed(n_i)
    
    # set up camera and render properties
    setupCamera(distance=6.15)
    
    ### generate surface
    rand_seed = np.random.randint(low = 0, high = num_surfaces)
#    generateRandomSpheres(20, 1, 1.5)
    create_landscape(size = terrain_size,subdivs=cloth_subdivs*2, seed = rand_seed)
        
    setupCloth(subdivs=cloth_subdivs, 
        size=cloth_size, 
        height=drop_height, 
        frame_end=last_frame,
        use_image_textures=use_image_textures,)
#        seed = rand_seed)
        
    ### save depth map
    # sets filepath to save to
    bpy.context.scene.render.filepath = save_path + "//depth_maps//" + "depth_" + str(n_i)
    
    # set up compositing for depth map
    setupCompositing('depth_map')
    
    # renders and saves image
    bpy.ops.render.render(write_still=1)
    
    depth_path = save_path + "//depth_maps//" + "depth_" + str(n_i)+'.tif' # saved in the data table
    
    print("depth map " + str(n_i) + " saved.")
    
    rand_light = np.random.randint(low = 0, high = light_conds)
    setupLight((parameters.light_posns_x[rand_light], parameters.light_posns_y[rand_light], parameters.light_posns_z[rand_light]))
        
    texture_img_path = Texture_folder + '//' + str(np.random.randint(low = 0, high = texture_types)) + '//texture_' + str(np.random.randint(low = 0, high = num_surfaces*light_conds)) + '.png'
    applyTexture(texture_img_path)
    # applyTexture(textures[np.random.randint(low = 0, high = texture_types)][np.random.randint(low = 0, high = num_surfaces*light_conds)])
    
#    setAmbient(parameters.ambient_intensity_levels[rand_light])
    setAmbient(0.9)
    
    # render and save the image
    image_file = saveImage(save_path, n_i)
            
    # save data in table
    #rowdict = {'surf':surf, 'light_type': light, 'tex_type': type, 'depthmap_file': depth_path, 'image_file': image_file, 'texture_file':texture}
    #row = pd.DataFrame(rowdict, index=[0])

    #texture_table = pd.concat([texture_table, row], axis=0)
            
    texture_index = texture_index + 1
            
    cleanup() # delete current cloth and terrain
        
            
            
