######## Polka Dot Texture Generator #########
# Celine Aubuchon 2023

import matplotlib.pyplot as plt
import numpy.random as rnd
import numpy as np
from matplotlib.patches import Ellipse
import random
import pandas as pd
import math
import os
import sys

import parameters
# this re-loads the "parameters.py" Text every time
import importlib
importlib.reload(parameters)

def genPolkaTextureGrid(name='test', numTexels=500, texelSize=3.568, minAspectRatio=1, maxAspectRatio=1, maxRotation=0, minSizeRatio=1):
    num_texels_per_row = int(math.sqrt(numTexels))
    print(num_texels_per_row)
    xrange = 100
    yrange = xrange
    size = texelSize

    jitter_max = 0.5*((xrange/num_texels_per_row) - size)

    xs = np.linspace(0.5*(xrange/num_texels_per_row), xrange - 0.5*(xrange/num_texels_per_row), num_texels_per_row, endpoint=True)
    ys = xs

    coords = []
    # add small random jitter to each pair of coordinates
    for i in np.arange(num_texels_per_row):
        for j in np.arange(num_texels_per_row):
            x = xs[i]; y = ys[j]
            p = [x+np.random.uniform(-jitter_max, jitter_max, 1), y+np.random.uniform(-jitter_max, jitter_max, 1)]
            coords.append(p)

    # initialize dummy ells list
    ells = [Ellipse(xy=coords[i], width=size, height=size, angle=rnd.rand()*360)
            for i in range(numTexels)]
    
    # repopulate ells list with actual ells
    for i in range(numTexels):
        sz = random.uniform(minSizeRatio, 1.0)*size
        aspectRat = random.uniform(minAspectRatio,maxAspectRatio)
        ells[i] = Ellipse(xy=coords[i], width=aspectRat*sz, height=sz, angle=rnd.rand()*maxRotation)


    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_facecolor([0.7,0.7,0.7])
    
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(1)
        e.set_facecolor([0.3,0.3,0.3])

    ax.set_xlim(0, xrange)
    ax.set_ylim(0, yrange)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(bottom=False, labelbottom=False,
                left=False, labelleft=False)

    folder = "Textures/"+str(type)+'/'
    folder_exists = os.path.isdir(folder)

    # if folder doesn't exist create it.
    if not folder_exists:
        os.makedirs(folder)

    fig.savefig(folder+name+'.png', dpi=100, format='png', bbox_inches='tight', pad_inches=0)


def genPolkaTexture(name='test', numTexels=500, texelSize=3.568, minAspectRatio=1, maxAspectRatio=1, maxRotation=0, minSizeRatio=1):
    
    NUM = numTexels
    xrange = 100
    yrange = xrange
    size = texelSize

    
    xys = [rnd.rand(2)*xrange for i in range(NUM)]
    sizes = np.ones((1,NUM))[0]*size

    try_count = 0
    bad_place_count = 0
    i = 0
    # check collisions
    while i < NUM:
        collision = False
        curr_pos = rnd.rand(2)*xrange
        try_count = 0
        for p in xys[:i]:
            dist_sq = (curr_pos[0] - p[0])**2 + (curr_pos[1] - p[1])**2

            if(try_count > 100):
                bad_place_count = bad_place_count + 1
                i = i + 1
                break

            if dist_sq < (sizes[i])**2:
                collision = True
                try_count = try_count + 1
                break
        if(not collision):
            xys[i] = curr_pos
            i = i + 1
                
    #print('bad place count: ', bad_place_count)
    
    # initialize dummy ells list
    ells = [Ellipse(xy=xys[i], width=size, height=size, angle=rnd.rand()*360)
            for i in range(NUM)]
    
    # repopulate ells list with actual ells
    for i in range(NUM):
        sz = random.uniform(minSizeRatio, 1.0)*size
        aspectRat = random.uniform(minAspectRatio,maxAspectRatio)
        ells[i] = Ellipse(xy=xys[i], width=aspectRat*sz, height=sz, angle=rnd.rand()*maxRotation)


    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_facecolor([0.7,0.7,0.7])
    
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(1)
        e.set_facecolor([0.3,0.3,0.3])

    ax.set_xlim(0, xrange)
    ax.set_ylim(0, yrange)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(bottom=False, labelbottom=False,
                left=False, labelleft=False)

    folder = "Textures/"+str(type)+'/'
    folder_exists = os.path.isdir(folder)

    # if folder doesn't exist create it.
    if not folder_exists:
        os.makedirs(folder)

    fig.savefig(folder+name+'.png', dpi=100, format='png', bbox_inches='tight', pad_inches=0)

### generate textures #########################################################################################
# load parameters
num_surfaces = parameters.num_surfaces
light_levels = parameters.light_levels
rotation_levels = parameters.rotation_levels
min_aspect_levels = parameters.min_aspect_levels
max_aspect_levels = parameters.max_aspect_levels
min_size_levels = parameters.min_size_levels
num_texels = parameters.num_texels
texel_size = parameters.texel_size

num_textures = num_surfaces * light_levels * len(rotation_levels)
columns = ['name', 'numTexels', 'type', 'maxRotation', 'minAspectRatio', 'minSizeRatio', 'texelSize']
texture_table = pd.DataFrame(columns=columns)


for type in np.arange(len(rotation_levels)):
    i = 0

    while i < math.ceil(num_textures/len(rotation_levels)):
        name = 'texture_'+str(i)
        genPolkaTextureGrid(name=name, numTexels=num_texels, maxRotation=rotation_levels[type], minAspectRatio=min_aspect_levels[type], maxAspectRatio=max_aspect_levels[type], minSizeRatio=min_size_levels[type])
        
        rowdict = {'name':name, 'numTexels':num_texels, 'type':type, 'maxRotation':rotation_levels[type], 'minAspectRatio':min_aspect_levels[type], 'minSizeRatio':min_size_levels[type], 'texelSize':texel_size}
        row = pd.DataFrame(rowdict, index=[0])

        texture_table = pd.concat([texture_table, row], axis=0)

        i = i + 1

texture_table.to_csv('texture_info.csv', index=False)