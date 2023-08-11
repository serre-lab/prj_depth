READ ME // Texture + Shading Landscape Stimuli
    Celine Aubuchon 2023


This pipeline creates images of landscape surfaces with different qualites of 
texture and shading. Most parameters necessary to change the number of surfaces
and qualities of shading and texture can be edited in the file 'parameters.py'.

Below are instructions on how to generate the images:

1) Change the parameters to reflect how many surfaces, texture qualities, and
shading qualities you want in 'parameters.py'.
2) Run the script polka_generator.py to generate unique textures of the different
qualities for each image (a new texture is used for every render). TAKES A WHILE
3) Open the blender file 'stim-generator.blend' and run the attatched script
'stim-generator-script.py'. There are settings within this file that are not
set in the script, so running it in a new blender file will not give the same 
effect. 

Below is a description of what each file does:

'parameters.py': Sets some relevant parameters for generating textures and images
    of textured and shaded surface, used both in 'polka_generator.py' and the blend
    file. See the comments in the file for what each parameter does.

'polka_generator': Creates and saves polka dot image textures with varying levels of 
    quality set by the parameters. The highest quality texture type consists of perfect 
    circular polka dots with a fixed size, since it yields compelling depth percepts
    when applied to curved surfaces. To decrease the quality, you can decrease the 
    aspect ratio of the texels while adding random rotation to create ellipse textures.
    The change in aspect ratio and rotation disrupts the foreshortening gradient that
    isotrophic textures produce on slanted/curved 3D surfaces. The relative size of 
    the texels can also be changed. Each texture type is saved into a subfolder of 
    'Textures/' that correspends to the texture type index. 

'stim-generator.blend'/'stim-generator-script.py': Generates the surfaces and renders
    depth maps and images of each surface under the different lighting and texture 
    qualities set in the parameters. To create one surface, it using the ANT Landscape
    generator plug-in and sets a unique seed specified cy the surface index. Because of
    this, regenerating the surfaces will produce the same geometries. 
    
    To change the quality of the lighting, both the world illumination (ambient light) 
    and position of a single point light are changed. High quality lighting produces high 
    contrast shadows by placing the point light so that it is oblique to the surface. 
    Lower quality lighting is created by moving the point light so that is it more above 
    the surface.

    To apply the texture to the surface, a 'stiff cloth' is dropped onto the surface and 
    an image texture is applied to it. This avoids the problem of uneven texture 'stretching'
    on the surface and replaces it with more natural looking folds. The cloth drop is simulated 
    using blender's physics engine. 

    Relative depth maps are rendered and saved as 'TIFF' images. To change this, you will need
    to edit the functions in the script. There are also some residule functiosn in the script 
    that are not used currently, that you may want to modify/incorperate. 

Direct any questions to celine.d.aubuchon@gmail.com.