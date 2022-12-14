from ursina import *
import cube_env


# modified from StanislavPetrovV's original code
# (https://github.com/StanislavPetrovV/Rubiks-Cube-3D)
class Game(Ursina):
    def __init__(self, seq=(0, 0, 0, 0), seq_str=None, state=None):
        super().__init__()
        window.fullscreen = False
        Entity(model='quad', scale=80, texture='white_cube', texture_scale=(80, 80), rotation_x=90, y=-5,
               color=color.dark_gray)  # plane
        EditorCamera()
        camera.world_position = (11.5, 13, -20)
        camera.world_rotation_y = -30
        camera.world_rotation_x = 30

        self.model, self.texture = 'models/custom_cube', 'textures/rubik_texture'
        self.seq, self.seq_str = seq, seq_str
        self.state = state
        self.load_game()
        button_scramble = Button(position=(0, -0.3), text='solve')
        button_scramble.fit_to_text()
        button_scramble.on_click = self.solve  # assign a function to the button.

    def load_game(self):
        self.create_cube_positions()
        self.CUBES = [Entity(model=self.model, texture=self.texture, position=pos) for pos in self.SIDE_POSITIONS]
        self.PARENT = Entity()
        self.rotation_axes = {'L': 'x', 'R': 'x', 'U': 'y', 'D': 'y', 'F': 'z', 'B': 'z'}

        self.cubes_side_positons = {'L': self.LEFT, 'D': self.DOWN, 'R': self.RIGHT, 'F': self.FRONT,
                                    'B': self.BACK, 'U': self.UP}

        if self.state is not None:
            self.initialize_with_surface()

        self.animation_time = 0.4
        self.action_trigger = True
        self.action_mode = False
        self.animation_mode = False
        self.message1 = Text(self.seq_str, origin=(0, 10), color=color.white)

        self.create_sensors()
        self.scramble(self.seq)  # initial state of the cube, rotations - number of side turns

    def initialize_with_surface(self):
        self.reparent_to_scene()
        print(self.state.edge_orientation)
        print(self.state.corner_orientation)
        print(self.state.edge_permutation)
        print(self.state.corner_permutation)
        self.CUBES[2].position = (-1, 1, -1)
        self.CUBES[2].rotation = (-90,90,0)

        corner_location = [(-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1),
                           (-1, -1, 1), (1, -1, 1), (1, -1, -1), (-1, -1, -1)]
        edge_location = [(0, 1, 1), (1, 1, 0), (0, 1, -1), (-1, 1, 0),
                         (-1, 0, 1), (1, 0, 1), (1, 0, -1), (-1, 0, -1),
                         (0, -1, 1), (1, -1, 0), (0, -1, -1), (-1, -1, 0)]
        if self.CUBES[i].position in corner_location:
            corner_location.index[self.CUBES[i].position]
        else:
            edge_location.index[self.CUBES[i].position]


        # self.edge_orientation = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]    # 0 to 2
        # self.edge_permutation = [YO, YG, YR, YB, OB, OG, RG, RB, WR, WG, WO, WB]    # 0 to 11
        # self.corner_orientation = [0, 0, 0, 0, 0, 0, 0, 0]              # 0 to 3
        # self.corner_permutation = [YBO, YOG, YRB, YRB, WBR, WRG, WGO, WOB]              # 0 to 7
        #
        # sum(edge_orientation) mod 2 = 0
        # sum(corner_orientation) mod 3 = 0

        #              c00 e00 c01
        #              e03 [U] e01   # expanded view
        #              c03 e02 c02
        #              c03 e02 c02
        # c00 e03 c03  c03 e02 c02  c02 e01 c01  c01 e00 c00
        # e04 [L] e07  e07 [F] e06  e06 [R] e05  e05 [B] e04
        # c07 e11 c04  c04 e08 c05  c05 e09 c06  c06 e10 c07
        #              c04 e08 c05
        #              e11 [D] e09
        #              c07 e10 c06

        #             c00 e00 c01  # layer view
        #            e03 [U] e01
        #           c03 e02 c02  : 3rd layer

        #             e04 [B] c05
        #            [L] [2] [R]
        #           e07 [F] c06  : 2nd layer

        #             c07 e10 c06
        #            e11 [D] e09
        #           c04 e08 c05  : 1st layer


    def solve(self):
        self.message2 = Text(text="Please wait...", origin=(0, -10), color=color.white)
        self.animation_mode = True
        seq_inv = cube_env.invseq(self.seq)
        self.scramble(seq_inv)

    def scramble(self, sequence):
        s = Sequence(2)
        for seq in sequence:
            side_name = 'UDFBRL'[seq % 6]
            rotation_angle = (90, -90, 180)[int(seq/6)]
            if side_name in 'LDB':
                rotation_angle *= -1
            s.append(Func(self.rotate_side, side_name, rotation_angle, self.animation_mode))
            if self.animation_mode:
                s.append(self.animation_time*abs(rotation_angle/90)+0.05)

        s.start()

    def rotate_side(self, side_name, rotation_angle=90, animation=True):
        self.action_trigger = False
        cube_positions = self.cubes_side_positons[side_name]
        rotation_axis = self.rotation_axes[side_name]
        self.reparent_to_scene()
        for cube in self.CUBES:
            if cube.position in cube_positions:
                cube.parent = self.PARENT
                if animation:
                    if rotation_axis == 'x':
                        self.PARENT.animate_rotation_x(rotation_angle, duration=self.animation_time*abs(rotation_angle/90))
                    if rotation_axis == 'y':
                        self.PARENT.animate_rotation_y(rotation_angle, duration=self.animation_time*abs(rotation_angle/90))
                    if rotation_axis == 'z':
                        self.PARENT.animate_rotation_z(rotation_angle, duration=self.animation_time*abs(rotation_angle/90))
                else:
                    if rotation_axis == 'x':
                        self.PARENT.rotation_x = rotation_angle
                    elif rotation_axis == 'y':
                        self.PARENT.rotation_y = rotation_angle
                    elif rotation_axis == 'z':
                        self.PARENT.rotation_z = rotation_angle

        invoke(self.toggle_animation_trigger, delay=self.animation_time*abs(rotation_angle/90) + 0.11)


    def create_sensors(self):
        '''detectors for each side, for detecting collisions with mouse clicks'''
        create_sensor = lambda name, pos, scale: Entity(name=name, position=pos, model='cube', color=color.dark_gray,
                                                        scale=scale, collider='box', visible=False)
        self.LEFT_sensor = create_sensor(name='L', pos=(-0.99, 0, 0), scale=(1.01, 3.01, 3.01))
        self.FRONT_sensor = create_sensor(name='F', pos=(0, 0, -0.99), scale=(3.01, 3.01, 1.01))
        self.BACK_sensor = create_sensor(name='B', pos=(0, 0, 0.99), scale=(3.01, 3.01, 1.01))
        self.RIGHT_sensor = create_sensor(name='R', pos=(0.99, 0, 0), scale=(1.01, 3.01, 3.01))
        self.UP_sensor = create_sensor(name='U', pos=(0, 1, 0), scale=(3.01, 1.01, 3.01))
        self.DOWN_sensor = create_sensor(name='D', pos=(0, -1, 0), scale=(3.01, 1.01, 3.01))

    def toggle_game_mode(self):
        '''switching view mode or interacting with Rubik's cube'''
        self.action_mode = not self.action_mode
        msg = dedent(f"{'ACTION mode ON' if self.action_mode else 'VIEW mode ON'}"
                     f" (to switch - press middle mouse button)").strip()
        # self.message2.text = msg

    def toggle_animation_trigger(self):
        '''prohibiting side rotation during rotation animation'''
        self.action_trigger = not self.action_trigger

    def reparent_to_scene(self):
        for cube in self.CUBES:
            if cube.parent == self.PARENT:
                world_pos, world_rot = round(cube.world_position, 1), cube.world_rotation
                cube.parent = scene
                cube.position, cube.rotation = world_pos, world_rot
        self.PARENT.rotation = 0

    def create_cube_positions(self):
        self.LEFT = {Vec3(-1, y, z) for y in range(-1, 2) for z in range(-1, 2)}
        self.DOWN = {Vec3(x, -1, z) for x in range(-1, 2) for z in range(-1, 2)}
        self.FRONT = {Vec3(x, y, -1) for x in range(-1, 2) for y in range(-1, 2)}
        self.BACK = {Vec3(x, y, 1) for x in range(-1, 2) for y in range(-1, 2)}
        self.RIGHT = {Vec3(1, y, z) for y in range(-1, 2) for z in range(-1, 2)}
        self.UP = {Vec3(x, 1, z) for x in range(-1, 2) for z in range(-1, 2)}
        self.SIDE_POSITIONS = self.LEFT | self.DOWN | self.FRONT | self.BACK | self.RIGHT | self.UP


    def input(self, key, israw=False):
        if key in 'mouse1 mouse3' and self.action_mode and self.action_trigger:
            for hitinfo in mouse.collisions:
                collider_name = hitinfo.entity.name
                if (key == 'mouse1' and collider_name in 'LRFB' or
                        key == 'mouse3' and collider_name in 'UD'):
                    if collider_name in 'URF':
                        rotation_angle = 90
                    else:
                        rotation_angle = -90
                    self.rotate_side(collider_name, rotation_angle)
                    self.message1.text += collider_name

                    break
        if key == 'mouse2':
            self.toggle_game_mode()
        super().input(key)


if __name__ == '__main__':
    game = Game()
    game.run()
