import taichi as ti
import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# * parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--iters', type=int, default=100)
parser.add_argument('--steps', type=int, default=4096)
options = parser.parse_args()

# * taichi settings
real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True, debug=True)

# * simulation settings
dim = 2
n_particles = 8192
inv_n_particles = 1 / n_particles
n_solid_particles = 0
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 5e-4
p_vol = 1
E = 20

# * baffles settings
# L = 0.4 # length of the baffle.
# H = 0.7 # height of the baffle.
# theta = math.pi / 3 # angle of the baffle.
left_endpoint = [0.3, 0.7]
right_endpoint = [0.7, 0.3]
baffle_dir = ti.Vector([right_endpoint[0] - left_endpoint[0], right_endpoint[1] - left_endpoint[1]])
normal = ti.Vector([- right_endpoint[1] + left_endpoint[1], - left_endpoint[0] + right_endpoint[0]])
baffle_dir /= baffle_dir.norm()
normal /= normal.norm()

inv_L = 1 / (baffle_dir.norm() + 1e-10)
target_left = 0.7
target_right = 0.9
target_mid = (target_left + target_right) / 2
target_height = 0.2

# TODO: update
mu = E
la = E
max_steps = 8192
max_itrs = options.iters
steps = options.steps
gravity = 3.8

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

particle_type = ti.field(ti.i32)
x, v = vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

# optimization variables
loss = scalar()
v_init = vec()

def allocate_fields():
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.i, n_particles).place(particle_type)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.place(loss, v_init)

    ti.root.lazy_grad()


@ti.kernel
def clear_grid():
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0, 0]
        grid_m_in[i, j] = 0
        grid_v_in.grad[i, j] = [0, 0]
        grid_m_in.grad[i, j] = 0
        grid_v_out.grad[i, j] = [0, 0]


@ti.kernel
def clear_particle_grad():
    # for all time steps and all particles
    for f, i in x:
        x.grad[f, i] = [0, 0]
        v.grad[f, i] = [0, 0]
        C.grad[f, i] = [[0, 0], [0, 0]]
        F.grad[f, i] = [[0, 0], [0, 0]]


@ti.kernel
def clear_particle_grad():
    for f, i in x:
        x.grad[f, i] = [0, 0]
        v.grad[f, i] = [0, 0]
        C.grad[f, i] = [[0, 0], [0, 0]]
        F.grad[f, i] = [[0, 0], [0, 0]]


def clear_grad():
    clear_particle_grad()
    clear_particle_grad()
    loss[None] = 0
    v_init.grad[None] = [0, 0]


@ti.kernel
def p2g(f: ti.i32):
    # input: step f
    # * APIC transfer
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        J = (new_F).determinant()
        if particle_type[p] == 0:  # fluid
            sqrtJ = ti.sqrt(J)
            new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])

        F[f + 1, p] = new_F
        r, s = ti.polar_decompose(new_F)

        cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        mass = 0.0
        if particle_type[p] == 0: # fluid
            mass = 4
            cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * E
        else: # solid
            mass = 1
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                     ti.Matrix.diag(2, la * (J - 1) * J)
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * C[f, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_in[base +
                          offset] += weight * (mass * v[f, p] + affine @ dpos)
                grid_m_in[base + offset] += weight * mass


bound = 4
bound_baffle = 2
coeff = 0.9

@ti.kernel
def grid_op():
    for i, j in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = inv_m * grid_v_in[i, j]
        v_out[1] -= dt * gravity

        # boundary conditions: velocity is zero when it is moving into the wall
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
            v_out[1] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
            v_out[1] = 0
        if j < bound and v_out[1] < 0:
            v_out[0] = 0
            v_out[1] = 0
        if j > n_grid - bound and v_out[1] > 0:
            v_out[0] = 0
            v_out[1] = 0


        # baffle1: box with two vertical walls
        if j < target_height * inv_dx:
            # -> left wall <- and -> right wall <-
            if (i > target_left * inv_dx - bound_baffle and i < target_left * inv_dx and v_out[0] > 0) or \
            (i > target_left * inv_dx and i < target_left * inv_dx + bound_baffle and v_out[0] < 0) or \
            (i > target_right * inv_dx - bound_baffle and i < target_right * inv_dx and v_out[0] > 0) or \
            (i > target_right * inv_dx and i < target_right * inv_dx + bound_baffle and v_out[0] < 0 ):
                v_out[0] = 0

        
        # baffle2: sloping side panels
        # ! Apply impulse reflection, may not be differentiable
        
        # ! method1: impulse reflection
        # if i < right_endpoint[0] * inv_dx:
        #     dist_p2l =((right_endpoint[0] - left_endpoint[0]) * (left_endpoint[1] - j * dx) - \
        #                 (right_endpoint[1] - left_endpoint[1]) * (left_endpoint[0] - i * dx)) * inv_L            
        #     if dist_p2l < bound_baffle * dx and dist_p2l > -bound_baffle * dx:
        #         lin = v_out.dot(normal)

        #         if (lin < 0 and dist_p2l < 0) or (lin > 0 and dist_p2l > 0):
        #             vit = v_out - lin * normal
        #             lit = vit.norm() + 1e-10
        #             if lit + coeff * lin <= 0:
        #                 v_out[0] = 0
        #                 v_out[1] = 0
        #             else:
        #                 v_out = (1 + coeff * lin / lit) * vit

        # ! method2: reset velocity to 0
        # if i > left_endpoint[0] * inv_dx and i < right_endpoint[0] * inv_dx:
        #     dist_p2l =((right_endpoint[0] - left_endpoint[0]) * (left_endpoint[1] - j * dx) - \
        #                 (right_endpoint[1] - left_endpoint[1]) * (left_endpoint[0] - i * dx)) * inv_L            
        #     if dist_p2l < bound_baffle * dx and dist_p2l > -bound_baffle * dx:
        #         v_out = [0, 0]

        # ! method3: impulse reflection, but ignore the friction
        # if i < right_endpoint[0] * inv_dx:
        #     dist_p2l =((right_endpoint[0] - left_endpoint[0]) * (left_endpoint[1] - j * dx) - \
        #                 (right_endpoint[1] - left_endpoint[1]) * (left_endpoint[0] - i * dx)) * inv_L            
        #     if dist_p2l < bound_baffle * dx and dist_p2l > -bound_baffle * dx:
        #         # v_out += - v_out
                # v_out += - 1.0 * v_out.dot(normal) * normal
                # v_out += - 1.0 * v_out.dot(baffle_dir) * baffle_dir
                # v_out = v_out.dot(baffle_dir) * baffle_dir
                # print('v_out', v_out)
                # assert False

        grid_v_out[i, j] = v_out


@ti.kernel
def g2p(f: ti.i32):
    # input: step f
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2] # Quadratic B-spline
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), real) - fx
                g_v = grid_v_out[base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C


@ti.ad.grad_replaced
def advance(s):
    clear_grid()
    p2g(s)
    grid_op()
    g2p(s)


@ti.ad.grad_for(advance)
def advance_grad(s):
    clear_grid()
    p2g(s)
    grid_op()

    g2p.grad(s) # ! after this step, the grad of x[s, p] will be reset as 0. But why?
    grid_op.grad()
    p2g.grad(s)


@ti.kernel
def compute_loss():
    for p in range(n_particles):
        if x[steps - 1, p][0] > 0 and x[steps - 1, p][0] < 1:
            loss[None] += (x[steps - 1, p][0] - target_mid) ** 2 * 1E-8


@ti.kernel
def initialize_v():
    for p in range(n_particles):
        v[0, p] = v_init[None]


def forward(total_steps=steps):
    # initialize
    initialize_v()

    # simulation
    for s in range(total_steps - 1):
        advance(s)

    # compute loss
    loss[None] = 0
    compute_loss()


class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = [] # particle positions
        self.particle_type = [] # 0: fluid, 1: solid
        self.offset_x = 0
        self.offset_y = 0

    def add_rect(self, x, y, w, h, ptype=1):
        # add a rectangle of particles
        global n_particles
        w_count = int(w / dx) * 2
        h_count = int(h / dx) * 2
        real_dx = w / w_count
        real_dy = h / h_count
        for i in range(w_count):
            for j in range(h_count):
                self.x.append([
                    x + (i + 0.5) * real_dx + self.offset_x,
                    y + (j + 0.5) * real_dy + self.offset_y
                ])
                self.particle_type.append(ptype)
                self.n_particles += 1
                self.n_solid_particles += int(ptype == 1)

    def set_offset(self, x, y):
        self.offset_x = x
        self.offset_y = y

    def finalize(self):
        global n_particles, n_solid_particles
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)


def build_scene(scene):
    scene.add_rect(0.1, 0.7, 0.2, 0.2, ptype=0)


gui = ti.GUI("Differentiable MPM", (640, 640), background_color=0xFFFFFF)


def visualize(s, folder):
    # visualize particles
    particles = x.to_numpy()[s]
    gui.circles(pos=particles, color=0x0, radius=1.5)

    # visualize boundaries
    delta_H = 3 * dx
    gui.line((0.05, delta_H), (0.95, delta_H), radius=4, color=0x0)

    # visualize baffle
    # gui.line(left_endpoint, right_endpoint, radius=4, color=0x0)
    gui.line((target_left, delta_H), (target_left, target_height), radius=4, color=0x0)
    gui.line((target_right, delta_H), (target_right, target_height), radius=4, color=0x0)

    os.makedirs(folder, exist_ok=True)
    gui.show(f'{folder}/{s:04d}.png')


def main():


    # initialize scene
    scene = Scene()
    build_scene(scene)
    scene.finalize()
    allocate_fields()

    v_init[None] = [0, 0]
    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        v[0, i] = [0, 0]
        F[0, i] = [[1, 0], [0, 1]]
        particle_type[i] = scene.particle_type[i]

    losses = []
    # training
    for iter in range(max_itrs):
        clear_grad()
        with ti.ad.Tape(loss, validation=True):
            forward()
        
        l = loss[None]
        losses.append(l)
        print('i=', iter, 'loss=', l)
        learning_rate = 1E-2

        print(v_init.grad[None])
        if not np.isnan(v_init.grad[None][0]) and not np.isnan(v_init.grad[None][1]) and not abs(v_init.grad[None][0]) > 2E2 and not abs(v_init.grad[None][1]) > 2E2:
            v_init[None] -= learning_rate * v_init.grad[None]
            v_init[None][0] = min(5, max(0, v_init[None][0]))
            v_init[None][1] = 0
        print(v_init[None])

        if iter % 20 == 0:
            # visualize
            forward(steps)
            for s in range(15, steps, 16):
                visualize(s, 'diffmpm/iter{:03d}/'.format(iter))


    # ti.profiler_print()
    plt.title("Optimization of Initial Velocity")
    plt.ylabel("Loss")
    plt.xlabel("Gradient Descent Iterations")
    plt.plot(losses)
    plt.show()
    # plt.savefig('exp1/loss/fluid_{:d}.png'.format(steps))


if __name__ == '__main__':
    main()
