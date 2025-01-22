import taichi as ti
import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt

real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True)

dim = 2
n_particles = 8192
n_solid_particles = 0
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_vol = 1
E = 20

# baffles
L = 0.4 # length of the baffle.
inv_L = 1 / L
target_left = 0.7
target_right = 0.9
target_height = 0.2

# TODO: update
mu = E
la = E
max_steps = 4096
steps = 4096
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
H = scalar()
theta = scalar()
loss = scalar()

def allocate_fields():
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.i, n_particles).place(particle_type)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.place(loss, H, theta)

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
bound_baffle = 2.5
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
            normal = ti.Vector([0.0, 1.0])
            lsq = (normal**2).sum()
            if lsq > 0.5:
                if ti.static(coeff < 0):
                    v_out[0] = 0
                    v_out[1] = 0
                else:
                    lin = v_out.dot(normal)
                    if lin < 0:
                        vit = v_out - lin * normal
                        lit = vit.norm() + 1e-10
                        if lit + coeff * lin <= 0:
                            v_out[0] = 0
                            v_out[1] = 0
                        else:
                            v_out = (1 + coeff * lin / lit) * vit
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
        left_endpoint = [0, H[None]]
        right_endpoint = [L * ti.sin(theta[None]), H[None] - L * ti.cos(theta[None])]
        if i < right_endpoint[0] * inv_dx:
            dist_p2l =((right_endpoint[0] - left_endpoint[0]) * (left_endpoint[1] - j * dx) - \
                        (right_endpoint[1] - left_endpoint[1]) * (left_endpoint[0] - i * dx)) * inv_L            
            if dist_p2l < bound_baffle * dx and dist_p2l > -bound_baffle * dx:
                normal = ti.Vector([- right_endpoint[1] + left_endpoint[1], - left_endpoint[0] + right_endpoint[0]])
                normal /= normal.norm()
                lin = v_out.dot(normal)

                if (lin < 0 and dist_p2l < 0) or (lin > 0 and dist_p2l > 0):
                    vit = v_out - lin * normal
                    lit = vit.norm() + 1e-10
                    if lit + coeff * lin <= 0:
                        v_out[0] = 0
                        v_out[1] = 0
                    else:
                        v_out = (1 + coeff * lin / lit) * vit

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

# @ti.ad.grad_replaced
def advance(s):
    clear_grid()
    p2g(s)
    grid_op()
    g2p(s)


# @ti.ad.grad_for(advance)
# def advance_grad(s):
#     clear_grid()
#     p2g(s)
#     grid_op()

#     g2p.grad(s)
#     grid_op.grad()
#     p2g.grad(s)


def forward(total_steps=steps):
    # simulation
    for s in range(total_steps - 1):
        advance(s)

    # compute loss


class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = [] # particle positions
        self.particle_type = [] # 0: fluid, 1: solid
        self.offset_x = 0
        self.offset_y = 0

        self.H = 0
        self.theta = 0


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

    def initialize_baffle(self, h, t):
        self.H = h
        self.theta = t

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
    scene.initialize_baffle(0.7, math.pi / 3)


gui = ti.GUI("Differentiable MPM", (640, 640), background_color=0xFFFFFF)


def visualize(s, folder):
    # visualize particles
    particles = x.to_numpy()[s]
    gui.circles(pos=particles, color=0x0, radius=1.5)

    # visualize boundaries
    delta_H = 3 * dx
    gui.line((0.05, delta_H), (0.95, delta_H), radius=4, color=0x0)

    # visualize baffle
    gui.line((0, delta_H + H[None]), (L * math.sin(theta[None]), delta_H + H[None] - L * math.cos(theta[None])), radius=4, color=0x0)
    gui.line((target_left, delta_H), (target_left, target_height), radius=4, color=0x0)
    gui.line((target_right, delta_H), (target_right, target_height), radius=4, color=0x0)

    os.makedirs(folder, exist_ok=True)
    gui.show(f'{folder}/{s:04d}.png')


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--steps', type=int, default=4096)
    options = parser.parse_args()

    steps = options.steps
    iters = options.iters

    print('experiment settings:')
    print('steps', steps)
    print('iters', iters)

    # initialize scene
    scene = Scene()
    build_scene(scene)
    scene.finalize()
    allocate_fields()

    H[None] = scene.H
    theta[None] = scene.theta
    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = [[1, 0], [0, 1]]
        particle_type[i] = scene.particle_type[i]

    losses = []

    # training
    for iter in range(iters):
        # with ti.ad.Tape(loss):
        # forward()
        # l = loss[None]
        # losses.append(l)
        # print('i=', iter, 'loss=', l)
        # learning_rate = 0.1

        # if iter % 10 == 0:
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
