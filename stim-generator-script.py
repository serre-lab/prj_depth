import bpy
import math
import os 
import sys
import numpy as np

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)
import parameters
# this re-loads the "parameters.py" Text every time
import importlib
importlib.reload(parameters)

### Functions ###
def setupCamera(distance):
# Set the camera object to the correct distance and rotation
    
    camera = bpy.data.objects['Camera']
    
    camera.location = (0, 0, distance)
    camera.rotation_euler = (0, 0, 0)
    
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
    
def setupLight(pos):
    light = bpy.data.objects.get('Light')#.select_set(True)
    light.location = pos
    
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
    
  
def applyTexture(image_path):
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
        tree.links.new(renderLayers_node.outputs["Depth"], normalize_node.inputs["Value"])
        tree.links.new(normalize_node.outputs["Value"], composite_node.inputs["Image"])
        
    if image_type == "normal_map":
        tree.links.new(renderLayers_node.outputs["Normal"], composite_node.inputs["Image"])
        bpy.context.scene.view_settings.view_transform = 'Raw'
        
def saveImage(save_path, seed, type, texture, light):
# renders and saves image of current scene 

    # names the image as a string of the seed, hdr, and texture information
    image_name = 'surf_' + str(seed) + '_type_' + str(type)+ \
        '_texture_' + str(texture) + '_light_'+ str(light)+ '.tif'

    # sets save format to PNG
    scene = bpy.context.scene
    scene.render.image_settings.file_format='TIFF'
    
    # sets filepath to save to
    scene.render.filepath = save_path + image_name
    
    # set up compositing for stimulus
    setupCompositing('stimulus')
    
    # renders and saves image
    bpy.ops.render.render(write_still=1)
    
    print(image_name + " saved.")
    
    return save_path + image_name + '.tif'
        

def cleanup():
# delete the previous surface objects to get ready for the next
    bpy.data.objects['Cloth'].select_set(True)
    bpy.ops.object.delete()
    bpy.data.objects['Terrain'].select_set(True)
    bpy.ops.object.delete()
    
    
    
    
########### SCRIPT #########################################################

# Settings

# generation settings
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

Texture_folder = bpy.path.abspath("//Textures")
save_path =  bpy.path.abspath("//Rendered_Images//")


## create texture filepath dictionary
textures = {}
keys = np.arange(parameters.num_tex_types)

for i in keys:
    textures[i] = [os.path.join(Texture_folder + '//'+str(i)+'//', f) for f in os.listdir(Texture_folder + '//'+str(i)+'//')]


# set up camera and render properties
setupCamera(distance=5)
setRenderProperties(samples=parameters.samples, res=parameters.res)
setupSceneFloor()


#columns = ['surf', 'light_type', 'tex_type', 'depthmap_file', 'image_file', 'texture_file']
#data_table = pd.DataFrame(columns=columns)

# generate surface, save depth map, set up different lights and textures, render and save image
texture_index = 0
for surf in range(num_surfaces):
    
    ### generate surface
    setupRandTerrain(seed=surf, 
        size=terrain_size, 
        height=terrain_height, 
        noise_depth=noise_depth)
        
    setupCloth(subdivs=cloth_subdivs, 
        size=cloth_size, 
        height=drop_height, 
        frame_end=last_frame,
        use_image_textures=use_image_textures)
        
    ### save depth map
    # sets filepath to save to
    bpy.context.scene.render.filepath = save_path + "//depth_maps//" + "depth_" + str(surf)
    
    # set up compositing for depth map
    setupCompositing('depth_map')
    
    # renders and saves image
    bpy.ops.render.render(write_still=1)
    
    depth_path = save_path + "//depth_maps//" + "depth_" + str(surf)+'.tif' # saved in the data table
    
    print("depth map " + str(surf) + " saved.")
    
    
    for light in range(light_conds):
        setupLight((parameters.light_posns_x[light], parameters.light_posns_y[light], parameters.light_posns_z[light]))
        
        for type in range(texture_types):
            
            texture = textures[type][texture_index]
            applyTexture(texture)
    
            setAmbient(parameters.ambient_intensity_levels[light])
            # render and save the image
            image_file = saveImage(save_path, surf, type, texture_index, light)
            
            # save data in table
            #rowdict = {'surf':surf, 'light_type': light, 'tex_type': type, 'depthmap_file': depth_path, 'image_file': image_file, 'texture_file':texture}
            #row = pd.DataFrame(rowdict, index=[0])

            #texture_table = pd.concat([texture_table, row], axis=0)
            
        texture_index = texture_index + 1
            
    cleanup() # delete current cloth and terrain
        
            
            
