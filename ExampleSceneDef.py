import ray
from ImLite import Image
from ray import evenly_tile_triangles
from utils import read_obj_triangles, to_srgb8, vec, from_srgb8

from PIL import Image as PILImage
import numpy as np


class ExampleSceneDef(object):
    def __init__(self, camera, scene, lights):
        self.camera = camera
        self.scene = scene
        self.lights = lights

    def render(
        self,
        output_path=None,
        output_shape=None,
        gamma_correct=True,
        srgb_whitepoint=None,
    ):
        # importlib.reload(ray)
        if output_shape is None:
            output_shape = [128, 128]
        if srgb_whitepoint is None:
            srgb_whitepoint = 1.0
        pix = ray.render_image(
            self.camera,
            self.scene,
            self.lights,
            output_shape[1],
            output_shape[0],
        )
        im = None
        if gamma_correct:
            cam_img_ui8 = to_srgb8(pix / srgb_whitepoint)
            im = Image(pixels=cam_img_ui8)
        else:
            im = im = Image(pixels=pix)
        if output_path is None:
            return im
        else:
            im.writeToFile(output_path)
# HELLO TA'S!!! THIS IS OUR MAIN SCENE SETUP!!!! 
# happy reading
"""we define a lot of ray.Materials in here. 
    they are a combination of basic colors and texture mapping
    there are like 50+ distinct objects that we manually assigned custom colors or textures to
    and then the scene is just the sum of all these objects.
"""
def KDHScene(wide_angle=False):
    # COLOR
    tan = ray.Material(vec([0.7, 0.7, 0.4]), 0.6)
    # load texture image (put texture_test.jpg next to this file)

    table_tex = from_srgb8(
        np.array(PILImage.open("table_small.jpg").convert("RGB"))
    )
    table_textured = ray.Material(
        vec([1.0, 1.0, 1.0]),
        k_s=0.3,
        p=90,
        # k_m=0.3,
        texture=table_tex,
        texture_repeat=(1.0, 1.0),
    )

    tablecloth_tex = from_srgb8(
        np.array(PILImage.open("tablecloth_custom.png").convert("RGB"))
    )
    tablecloth_textured = ray.Material(
        vec([1.0, 1.0, 1.0]),
        k_s=0.6,
        p=45,
        texture=tablecloth_tex,
        texture_repeat=(1.0, 1.0),
    )

    # hand t exture
    hand_tex = from_srgb8(np.array(PILImage.open("hand.png").convert("RGB")))
    hand_textured = ray.Material(
        vec([1.0, 1.0, 1.0]),
        k_s=0.6,
        p=45,
        texture=hand_tex,
        texture_repeat=(1.0, 1.0),
    )

    rumi_ramen_tex = from_srgb8(
        np.array(PILImage.open("rumi_ramen.png").convert("RGB"))
    )
    rumi_ramen_textured = ray.Material(
        vec([1.0, 1.0, 1.0]),
        k_s=0.6,
        p=45,
        texture=rumi_ramen_tex,
        texture_repeat=(1.0, 1.0),
    )

    rumi_ramen_body_tex = from_srgb8(
        np.array(PILImage.open("rumi_ramen_body.png").convert("RGB"))
    )
    rumi_ramen_body_textured = ray.Material(
        vec([1.0, 1.0, 1.0]),
        k_s=0.6,
        p=45,
        texture=rumi_ramen_body_tex,
        texture_repeat=(1.0, 1.0),
    )

    mira_ramen_tex = from_srgb8(
        np.array(PILImage.open("mira_ramen.png").convert("RGB"))
    )
    mira_ramen_textured = ray.Material(
        vec([1.0, 1.0, 1.0]),
        k_s=0.6,
        p=45,
        texture=mira_ramen_tex,
        texture_repeat=(1.0, 1.0),
    )

    mira_ramen_body_tex = from_srgb8(
        np.array(PILImage.open("mira_ramen_body.png").convert("RGB"))
    )
    mira_ramen_body_textured = ray.Material(
        vec([1.0, 1.0, 1.0]),
        k_s=0.6,
        p=45,
        texture=mira_ramen_body_tex,
        texture_repeat=(1.0, 1.0),
    )

    zoey_ramen_tex = from_srgb8(
        np.array(PILImage.open("zoey_ramen.png").convert("RGB"))
    )
    zoey_ramen_textured = ray.Material(
        vec([1.0, 1.0, 1.0]),
        k_s=0.6,
        p=45,
        texture=zoey_ramen_tex,
        texture_repeat=(1.0, 1.0),
    )
    zoey_ramen_body_tex = from_srgb8(
        np.array(PILImage.open("zoey_ramen_body.png").convert("RGB"))
    )
    zoey_ramen_body_textured = ray.Material(
        vec([1.0, 1.0, 1.0]),
        k_s=0.6,
        p=45,
        texture=zoey_ramen_body_tex,
        texture_repeat=(1.0, 1.0),
    )

    plate_tex = from_srgb8(
        np.array(PILImage.open("plate_small.png").convert("RGB"))
    )
    plate_textured = ray.Material(
        vec([1.0, 1.0, 1.0]),
        k_s=0.6,
        p=45,
        texture=plate_tex,
        texture_repeat=(1.0, 1.0),
    )

    gray = ray.Material(vec([0.2, 0.2, 0.4]))
    green1 = ray.Material(vec([0.019, 0.371, 0.091]))

    blue1 = ray.Material(vec([0.045, 0.021, 0.558]))

    purple = ray.Material(vec([0.045, 0.021, 0.8]), 0.6)

    # balls
    skewer_color = ray.Material(vec([0.288, 0.221, 0.122]))
    ball_color = ray.Material(vec([0.996, 0.726, 0.658]))
    triangle_color = ray.Material(vec([0.861, 0.35, 0.12]))
    cube_color = ray.Material(vec([0.8, 0.3, 0.17]))

    # bowl color
    bowl_color = ray.Material(vec([0.5, 0.38, 0.28]))

    glass_color = ray.Material(
        vec([0.02, 0.02, 0.02]),
        k_s=0.3,
        p=90,
        k_m=0.1,
        k_a=vec([0.01, 0.01, 0.01]),
        k_t=0.85,
        ior=1.5,
    )
    metal_color = ray.Material(vec([0.7, 0.6, 0.6]), k_s=1.1, k_m=1)
    body_color = ray.Material(vec([0.26, 0.8, 0.17]))
    # bread
    bread_bowl_color = ray.Material(vec([0.4, 0.09, 0.1]), k_m=0.5)
    rolls_color = ray.Material(vec([0.93, 0.33, 0.064]))
    long_color = ray.Material(vec([0.405, 0.162, 0.064]))

    # ramen colors
    ramen_green = ray.Material(vec([0.047, 0.198, 0.064]))
    ramen_red = ray.Material(vec([0.323, 0.032, 0.042]))

    # kimbap
    seaweed = ray.Material(vec([0.03, 0.012, 0.0]))
    rice = ray.Material(vec([0.98, 1.0, 0.9]))
    carrot = ray.Material(vec([0.8, 0.1, 0.0]))
    kimbap_green = ray.Material(vec([0.08, 0.3, 0.06]))
    pickle = ray.Material(vec([1.0, 0.7, 0.1]))

    # sausage color
    sausage_color = ray.Material(vec([0.061, 0.019, 0.026]))

    # ramen colors
    ramen_green = ray.Material(vec([0.047, 0.198, 0.064]))
    ramen_red = ray.Material(vec([0.323, 0.032, 0.042]))

    # kimbap
    seaweed = ray.Material(vec([0.03, 0.012, 0.0]))
    rice = ray.Material(vec([0.98, 1.0, 0.9]))
    carrot = ray.Material(vec([0.8, 0.1, 0.0]))
    kimbap_green = ray.Material(vec([0.08, 0.3, 0.06]))
    pickle = ray.Material(vec([1.0, 0.7, 0.1]))

    # standing skewer colors
    skewer_stand_color = ray.Material(vec([0.02, 0.01, 0.01]))
    skewer_standing_color = ray.Material(vec([0.28, 0.22, 0.12]))
    fish_ball_color = ray.Material(vec([0.252, 0.04, 0.01]))
    fish_cake_1_color = ray.Material(vec([0.24, 0.14, 0.07]))
    fish_cake_2_color = ray.Material(vec([0.71, 0.28, 0.09]))

    table = read_obj_triangles(open("objs/table_yellow.obj"))
    tablecloth = read_obj_triangles(open("objs/tablecloth_blue_grey.obj"))
    plate_green_1 = read_obj_triangles(
        open("objs/plate_green_1.obj")
    )  # plate green 1 is the one at the top!!
    plate_green_2 = read_obj_triangles(open("objs/plate_green_2.obj"))

    # hand
    hand = read_obj_triangles(open("objs/hand.obj"))
    # ramen
    ramen_blue_bottom = read_obj_triangles(open("objs/ramen_blue_bottom.obj"))
    ramen_blue_body = read_obj_triangles(open("objs/ramen_blue_body.obj"))
    ramen_blue_lid = read_obj_triangles(open("objs/ramen_blue_lid.obj"))

    ramen_green_bottom = read_obj_triangles(open("objs/ramen_green_bottom.obj"))
    ramen_green_body = read_obj_triangles(open("objs/ramen_green_body.obj"))
    ramen_green_lid = read_obj_triangles(open("objs/ramen_green_lid.obj"))

    ramen_red_bottom = read_obj_triangles(open("objs/ramen_red_bottom.obj"))             
    ramen_red_body = read_obj_triangles(open("objs/ramen_red_body.obj"))
    ramen_red_lid = read_obj_triangles(open("objs/ramen_red_lid.obj"))

    # fish stick
    skewer = read_obj_triangles(open("objs/skewer.obj"))
    fish_ball = read_obj_triangles(open("objs/fish_ball.obj"))
    fish_triangle = read_obj_triangles(open("objs/fish_triangle.obj"))
    fish_cube = read_obj_triangles(open("objs/fish_cube.obj"))

    # standing fish stick
    stand = read_obj_triangles(open("objs/skewer_stand.obj"))
    standing_skewer = read_obj_triangles(open("objs/skewer_standing.obj"))
    fish_ball_standing = read_obj_triangles(open("objs/fish_ball_standing.obj"))
    fish_cake_standing_1 = read_obj_triangles(
        open("objs/fish_cake_standing_1.obj")
    )
    fish_cake_standing_2 = read_obj_triangles(
        open("objs/fish_cake_standing_2.obj")
    )

    # small bowl
    small_bowl = read_obj_triangles(open("objs/small_bowl.obj"))

    # soda
    soda_metal = read_obj_triangles(open("objs/soda_metal.obj"))
    soda_body = read_obj_triangles(open("objs/soda_body.obj"))

    # bread
    bread_bowl = read_obj_triangles(open("objs/bread_bowl.obj"))
    rolls = read_obj_triangles(open("objs/rolls.obj"))
    baguette = read_obj_triangles(open("objs/baguette.obj"))

    # kimbap!!
    long_kimbap_seaweed = read_obj_triangles(
        open("objs/long_kimbap_seaweed.obj")
    )
    long_kimbap_rice = read_obj_triangles(open("objs/long_kimbap_rice.obj"))
    long_kimbap_carrot = read_obj_triangles(open("objs/long_kimbap_carrot.obj"))
    long_kimbap_green = read_obj_triangles(open("objs/long_kimbap_green.obj"))
    long_kimbap_pickle = read_obj_triangles(open("objs/long_kimbap_pickle.obj"))

    # small kimbap
    small_kimbap_1_seaweed = read_obj_triangles(
        open("objs/small_kimbap_1_seaweed.obj")
    )
    small_kimbap_1_rice = read_obj_triangles(
        open("objs/small_kimbap_1_rice.obj")
    )
    small_kimbap_1_pickle = read_obj_triangles(
        open("objs/small_kimbap_1_pickle.obj")
    )
    small_kimbap_1_green = read_obj_triangles(
        open("objs/small_kimbap_1_green.obj")
    )
    small_kimbap_1_carrot = read_obj_triangles(
        open("objs/small_kimbap_1_carrot.obj")
    )

    small_kimbap_2_seaweed = read_obj_triangles(
        open("objs/small_kimbap_2_seaweed.obj")
    )
    small_kimbap_2_rice = read_obj_triangles(
        open("objs/small_kimbap_2_rice.obj")
    )
    small_kimbap_2_pickle = read_obj_triangles(
        open("objs/small_kimbap_2_pickle.obj")
    )
    small_kimbap_2_green = read_obj_triangles(
        open("objs/small_kimbap_2_green.obj")
    )
    small_kimbap_2_carrot = read_obj_triangles(
        open("objs/small_kimbap_2_carrot.obj")
    )

    small_kimbap_3_seaweed = read_obj_triangles(
        open("objs/small_kimbap_3_seaweed.obj")
    )
    small_kimbap_3_rice = read_obj_triangles(
        open("objs/small_kimbap_3_rice.obj")
    )
    small_kimbap_3_pickle = read_obj_triangles(
        open("objs/small_kimbap_3_pickle.obj")
    )
    small_kimbap_3_green = read_obj_triangles(
        open("objs/small_kimbap_3_green.obj")
    )
    small_kimbap_3_carrot = read_obj_triangles(
        open("objs/small_kimbap_3_carrot.obj")
    )

    small_kimbap_4_seaweed = read_obj_triangles(
        open("objs/small_kimbap_4_seaweed.obj")
    )
    small_kimbap_4_rice = read_obj_triangles(
        open("objs/small_kimbap_4_rice.obj")
    )
    small_kimbap_4_pickle = read_obj_triangles(
        open("objs/small_kimbap_4_pickle.obj")
    )
    small_kimbap_4_green = read_obj_triangles(
        open("objs/small_kimbap_4_green.obj")
    )
    small_kimbap_4_carrot = read_obj_triangles(
        open("objs/small_kimbap_4_carrot.obj")
    )

    small_kimbap_5_seaweed = read_obj_triangles(
        open("objs/small_kimbap_5_seaweed.obj")
    )
    small_kimbap_5_rice = read_obj_triangles(
        open("objs/small_kimbap_5_rice.obj")
    )
    small_kimbap_5_pickle = read_obj_triangles(
        open("objs/small_kimbap_5_pickle.obj")
    )
    small_kimbap_5_green = read_obj_triangles(
        open("objs/small_kimbap_5_green.obj")
    )
    small_kimbap_5_carrot = read_obj_triangles(
        open("objs/small_kimbap_5_carrot.obj")
    )

    table_tris = [ray.Triangle(vs, table_textured) for vs in table]
    evenly_tile_triangles(table_tris)

    tablecloth_tris = [
        ray.Triangle(vs, tablecloth_textured) for vs in tablecloth
    ]
    evenly_tile_triangles(tablecloth_tris)

    rumi_ramen_tris = [
        ray.Triangle(vs, rumi_ramen_textured) for vs in ramen_blue_lid
    ]
    evenly_tile_triangles(rumi_ramen_tris)

    rumi_ramen_body_tris = [
        ray.Triangle(vs, rumi_ramen_body_textured) for vs in ramen_blue_body
    ]
    evenly_tile_triangles(rumi_ramen_body_tris)

    mira_ramen_tris = [
        ray.Triangle(vs, mira_ramen_textured) for vs in ramen_red_lid
    ]
    evenly_tile_triangles(mira_ramen_tris)

    mira_ramen_body_tris = [
        ray.Triangle(vs, mira_ramen_body_textured) for vs in ramen_red_body
    ]
    evenly_tile_triangles(mira_ramen_body_tris)

    zoey_ramen_tris = [
        ray.Triangle(vs, zoey_ramen_textured) for vs in ramen_green_lid
    ]
    evenly_tile_triangles(zoey_ramen_tris)

    zoey_ramen_body_tris = [
        ray.Triangle(vs, zoey_ramen_textured) for vs in ramen_green_body
    ]
    evenly_tile_triangles(zoey_ramen_body_tris)

    plate_green_1_tris = [
        ray.Triangle(vs, plate_textured) for vs in plate_green_1
    ]
    evenly_tile_triangles(plate_green_1_tris)

    plate_green_2_tris = [
        ray.Triangle(vs, plate_textured) for vs in plate_green_2
    ]
    evenly_tile_triangles(plate_green_2_tris)

    hand_tris = [ray.Triangle(vs, hand_textured) for vs in hand]
    evenly_tile_triangles(hand_tris)
    # sausage
    sausage = read_obj_triangles(open("objs/sosig.obj"))

    scene = ray.Scene(
        table_tris
        + hand_tris
        + tablecloth_tris
        + [ray.Triangle(vs, skewer_stand_color) for vs in stand]
        + [ray.Triangle(vs, skewer_standing_color) for vs in standing_skewer]
        + [ray.Triangle(vs, fish_ball_color) for vs in fish_ball_standing]
        + [ray.Triangle(vs, fish_cake_1_color) for vs in fish_cake_standing_1]
        + [ray.Triangle(vs, fish_cake_2_color) for vs in fish_cake_standing_2]
        + plate_green_1_tris
        + plate_green_2_tris
        + [ray.Triangle(vs, sausage_color) for vs in sausage]
        + rumi_ramen_tris
        + rumi_ramen_body_tris
        + [ray.Triangle(vs, tan) for vs in ramen_blue_bottom]
        + [ray.Triangle(vs, skewer_color) for vs in skewer]
        + [ray.Triangle(vs, ball_color) for vs in fish_ball]
        + [ray.Triangle(vs, triangle_color) for vs in fish_triangle]
        + [ray.Triangle(vs, cube_color) for vs in fish_cube]
        + [ray.Triangle(vs, bread_bowl_color) for vs in bread_bowl]
        + [ray.Triangle(vs, rolls_color) for vs in rolls]
        + [ray.Triangle(vs, long_color) for vs in baguette]
        + [ray.Triangle(vs, ramen_green) for vs in ramen_green_body]
        + zoey_ramen_tris
        + zoey_ramen_body_tris
        + [ray.Triangle(vs, tan) for vs in ramen_green_bottom]
        + mira_ramen_body_tris
        + mira_ramen_tris
        + [ray.Triangle(vs, tan) for vs in ramen_red_bottom]
        + [ray.Triangle(vs, seaweed) for vs in long_kimbap_seaweed]
        + [ray.Triangle(vs, rice) for vs in long_kimbap_rice]
        + [ray.Triangle(vs, carrot) for vs in long_kimbap_carrot]
        + [ray.Triangle(vs, kimbap_green) for vs in long_kimbap_green]
        + [ray.Triangle(vs, pickle) for vs in long_kimbap_pickle]
        + [ray.Triangle(vs, seaweed) for vs in small_kimbap_1_seaweed]
        + [ray.Triangle(vs, rice) for vs in small_kimbap_1_rice]
        + [ray.Triangle(vs, carrot) for vs in small_kimbap_1_carrot]
        + [ray.Triangle(vs, kimbap_green) for vs in small_kimbap_1_green]
        + [ray.Triangle(vs, pickle) for vs in small_kimbap_1_pickle]
        + [ray.Triangle(vs, seaweed) for vs in small_kimbap_2_seaweed]
        + [ray.Triangle(vs, rice) for vs in small_kimbap_2_rice]
        + [ray.Triangle(vs, carrot) for vs in small_kimbap_2_carrot]
        + [ray.Triangle(vs, kimbap_green) for vs in small_kimbap_2_green]
        + [ray.Triangle(vs, pickle) for vs in small_kimbap_2_pickle]
        + [ray.Triangle(vs, seaweed) for vs in small_kimbap_3_seaweed]
        + [ray.Triangle(vs, rice) for vs in small_kimbap_3_rice]
        + [ray.Triangle(vs, carrot) for vs in small_kimbap_3_carrot]
        + [ray.Triangle(vs, kimbap_green) for vs in small_kimbap_3_green]
        + [ray.Triangle(vs, pickle) for vs in small_kimbap_3_pickle]
        + [ray.Triangle(vs, seaweed) for vs in small_kimbap_4_seaweed]
        + [ray.Triangle(vs, rice) for vs in small_kimbap_4_rice]
        + [ray.Triangle(vs, carrot) for vs in small_kimbap_4_carrot]
        + [ray.Triangle(vs, kimbap_green) for vs in small_kimbap_4_green]
        + [ray.Triangle(vs, pickle) for vs in small_kimbap_4_pickle]
        + [ray.Triangle(vs, seaweed) for vs in small_kimbap_5_seaweed]
        + [ray.Triangle(vs, rice) for vs in small_kimbap_5_rice]
        + [ray.Triangle(vs, carrot) for vs in small_kimbap_5_carrot]
        + [ray.Triangle(vs, kimbap_green) for vs in small_kimbap_5_green]
        + [ray.Triangle(vs, pickle) for vs in small_kimbap_5_pickle]
        + [ray.Sphere(vec([-2.4, 0.75, 0.25]), 1, glass_color)],
        bg_color=vec([0, 0, 0])
    )

    lights = [
        ray.PointLight(vec([2, 10, -5]), vec([100, 60, 40])),
        ray.PointLight(vec([-2, 10, -5]), vec([125, 75, 110])),
        ray.AmbientLight(0.2),
    ]

    if wide_angle:
        # camera = ray.Camera(
        #     vec([7, 4, 16]),
        #     target=vec([0.5, 0, -0.2]),
        #     vfov=15,
        #     aspect=16 / 9,
        # )
        camera = ray.Camera(
            vec([0.5, 8.5, 7]),
            target=vec([0.5, 0, -0.2]),
            up=vec([0.000001, 0.000001, -3]),
            vfov=27,
            aspect=16 / 9,
        )
    else:
        camera = ray.Camera(
            vec([0.5, 10, -0.2]),
            target=vec([0.5, 0, -0.2]),
            up=vec([0.000001, 0.000001, -3]),
            vfov=27,
            aspect=16 / 9,
        )

    return ExampleSceneDef(camera=camera, scene=scene, lights=lights)



def TwoSpheresExample():
    tan = ray.Material(vec([0.7, 0.7, 0.4]), 0.6)
    gray = ray.Material(vec([0.2, 0.2, 0.2]))

    scene = ray.Scene(
        [
            ray.Sphere(vec([0, 0, 0]), 0.5, tan),
            ray.Sphere(vec([0, -40, 0]), 39.5, gray),
        ]
    )

    lights = [
        ray.PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
        ray.AmbientLight(0.1),
    ]
    camera = ray.Camera(
        vec([3, 1.7, 5]), target=vec([0, 0, 0]), vfov=25, aspect=16 / 9
    )
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights)


def ThreeSpheresExample():
    # load texture image (put texture_test.jpg next to this file)
    tex = np.array(PILImage.open("texture_test.jpg").convert("RGB"))

    # create a textured material (this uses Material.sample to handle uint8 -> [0,1])
    textured = ray.Material(
        vec([1.0, 1.0, 1.0]),
        k_s=0.3,
        p=90,
        k_m=0.3,
        texture=tex,
        texture_repeat=(100.0, 100.0),
    )

    tan = ray.Material(vec([0.4, 0.4, 0.2]), k_s=0.3, p=90, k_m=0.3)
    blue = ray.Material(vec([0.2, 0.2, 0.5]), k_m=0.5)
    gray = ray.Material(vec([0.2, 0.2, 0.2]), k_m=0.4)

    scene = ray.Scene(
        [
            # left sphere uses textured material
            ray.Sphere(vec([-0.7, 0, 0]), 0.5, tan),
            ray.Sphere(vec([0.7, 0, 0]), 0.5, blue),
            ray.Sphere(vec([0, -40, 0]), 39.5, textured),
        ]
    )

    lights = [
        ray.PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
        ray.AmbientLight(0.1),
    ]

    camera = ray.Camera(
        vec([3, 1.2, 5]), target=vec([0, -0.4, 0]), vfov=24, aspect=16 / 9
    )
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights)


def CubeExample():
    tan = ray.Material(vec([0.7, 0.7, 0.4]), 0.6)
    _ = ray.Material(vec([0.2, 0.2, 0.2]))

    # Read the triangle mesh for a 2x2x2 cube, and scale it down to 1x1x1 to
    # fit the scene.
    vs_list = 0.5 * read_obj_triangles(open("cube.obj"))

    scene = ray.Scene(
        [
            # Make triangle objects from the vertex coordinates
            ray.Triangle(vs, tan)
            for vs in vs_list
        ]
    )

    lights = [
        ray.PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
        ray.AmbientLight(0.1),
    ]

    camera = ray.Camera(
        vec([3, 1.7, 5]), target=vec([0, 0, 0]), vfov=25, aspect=16 / 9
    )
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights)

def OrthoFriendlyExample(sphere_radius=0.25):
    gray = ray.Material(vec([0.5, 0.5, 0.5]))

    # One small sphere centered at z=-0.5
    scene = ray.Scene(
        [
            ray.Sphere(vec([0, 0, -0.5]), sphere_radius, gray),
        ]
    )

    lights = [
        ray.AmbientLight(0.5),
    ]
    camera = ray.Camera(
        vec([0, 0, 0]), target=vec([0, 0, -0.5]), vfov=90, aspect=1
    )
    return ExampleSceneDef(camera=camera, scene=scene, lights=lights)
