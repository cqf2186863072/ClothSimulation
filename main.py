import taichi as ti
ti.init(arch=ti.vulkan)  # Alternatively, ti.init(arch=ti.cpu)

#Set physics parameters
to_generate = False
to_reset = False

n = 128
quad_size = 1.0 / n
dt = 4e-2 / n
substeps = int(1 / 60 // dt)

gravity = ti.Vector([0, -9.8, 0])
spring_Y = ti.field(ti.f32, shape=())
spring_Y[None] = 1e3
dashpot_damping = 1e4
drag_damping = 1

ball_radius= ti.field(ti.f32, shape=())
ball_radius[None] = 0.3
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [0, 0, 0]

x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))

num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

bending_springs = False

def reset():
    substeps = int(1 / 60 // dt)
    spring_Y[None] = 1e3
    ball_radius[None] = 0.3

@ti.kernel
def initialize_mass_points():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1

    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.5 + random_offset[0], 0.6,
            j * quad_size - 0.5 + random_offset[1]
        ]
        v[i, j] = [0, 0, 0]


@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # 1st triangle of the square
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # 2nd triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * n + j] = (0.9, 0.9, 0.9)
        else:
            colors[i * n + j] = (0.1, 0.1, 0.1)

initialize_mesh_indices()

spring_offsets = []
if bending_springs:
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i, j) != (0, 0):
                spring_offsets.append(ti.Vector([i, j]))

else:
    for i in range(-2, 3):
        for j in range(-2, 3):
            if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                spring_offsets.append(ti.Vector([i, j]))

@ti.kernel
def substep():
    for i in ti.grouped(x):
        v[i] += gravity * dt

    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                d = x_ij.normalized()
                current_dist = x_ij.norm()
                original_dist = quad_size * float(i - j).norm()
                # Spring force
                force += -spring_Y[None] * d * (current_dist / original_dist - 1)
                # Dashpot damping
                force += -v_ij.dot(d) * d * dashpot_damping * quad_size

        v[i] += force * dt

    for i in ti.grouped(x):
        v[i] *= ti.exp(-drag_damping * dt)
        offset_to_center = x[i] - ball_center[0]
        if offset_to_center.norm() <= ball_radius[None]:
            # Velocity projection
            normal = offset_to_center.normalized()
            v[i] -= min(v[i].dot(normal), 0) * normal
        x[i] += dt * v[i]

@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]

#Set ui parameters
window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (768, 768),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0.6, 0.5, 0.4))
camera = ti.ui.Camera()
camera.position(0.0, 1.5, 3)
camera.lookat(0.0, 0.0, -1.5)
scene = ti.ui.Scene()
scene.set_camera(camera)
gui = window.get_gui()

while window.running:
    with gui.sub_window("Parameters", x=0, y=0, width=0.4, height=0.4):
        spring_Y[None] = gui.slider_float("spring_Y", spring_Y[None], minimum=1e2, maximum=3e4)
        substeps = gui.slider_int("substeps", substeps, minimum=10, maximum=100)
        ball_radius[None] = gui.slider_float("ball_radius", ball_radius[None], minimum=0.1, maximum=0.5)
        to_reset = gui.button("Reset")
        to_generate = gui.button("Generate")

    if to_generate:
        initialize_mass_points()
        is_ready = False
    
    if to_reset:
        reset()
        to_reset = False
    '''
    if window.get_event(ti.ui.PRESS):
        if window.is_pressed(ti.ui.RMB):
            initialize_mass_points()
    '''

    for i in range(substeps):
        substep()
    update_vertices()

    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))

    # Draw a smaller ball to avoid visual penetration
    scene.particles(ball_center, radius=ball_radius[None] * 0.95, color=(0.4, 0.5, 0.6))

    canvas.scene(scene)
    window.show()