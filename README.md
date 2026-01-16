# bruh python ray tracer

A CPU-based ray tracer in Python. Tried to render on multiple core and killed my potato laptop a few times but worth it. Initially we complainied because it would be inherently so slow but we realize it is much more easy to debug and do math. Therefore, thank god this project was in python. Has cool texture mapping, glass refraction, and BVH acceleration. Built for Cornell's CS 4620. Software engineering hates to see us coming. 

Also, this submission made the top grade in the class for this particular assignment!

## video


https://github.com/user-attachments/assets/208d5c23-cc23-437a-b86b-73b9d3753823




## what it do

**Backwards (Whitted-style) ray tracer** that shoots rays from camera through pixels into the scene, calculating mathematically realistic lighting + shadows, reflections, and refractions 

## Key Features

- **ray-sphere & ray-triangle intersection** (quadratic formula & Möller-Trumbore)
- **Lambertian diffuse + Blinn-Phong specular** algorithmic lighting with shadows
- **glass reflection** done with Snell's law
- **texture mapping** with bilinear filtering and automatic spherical UVs
- **Multi-core rendering:** 64×64 tiles via ProcessPoolExecutor
  - spawns multiple Python interpreters across CPU cores. This make significant speedup over single-threaded rendering (still took an hour vs 3ish hours and hogged all cpu processes litearlly it was 100% on task manager.) If Python GIL has one enemy it is me if python GIL has no enemies then i am dead 
- **BVH acceleration**, has fast intersection testing and very cool computer graphics thing. Tests closer child first so we can terminate early if we need



## files

- `ray.py` - Core ray tracing engine
- `ExampleSceneDef.py` - define the scene
- `objs/` - 3D models and textures. i 3d modelled all of them in blender!
- `wips/` - work in progress rendered outputs (cringe)

## use

```bash
python testray.py                           # Default 256×144
python testray.py --nx 1920 --ny 1080      # HD render
python testray.py --outFile render.png     # Custom output
```
