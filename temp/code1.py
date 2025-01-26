import taichi as ti
ti.init()

steps = 10
N = 100
x = ti.Vector.field(2, dtype=ti.f32, shape=(steps, N), needs_grad=True)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

nx = 3
ny = 4
normal = ti.Vector([nx, ny])
normal = normal.normalized()
direction = ti.Vector([-ny, nx])
direction = direction.normalized()

@ti.kernel
def func(s: ti.i32):
    for i in range(N):
        v_out = x[s, i]
        v_out = 1.0 * v_out.dot(normal) * normal # C10
        v_out += - 1.0 * v_out.dot(direction) * direction # C11
        x[s + 1, i] = v_out + [0.0, 100 * i / N]

@ti.kernel
def compute_loss():
    for i in range(N):
        loss[None] += x[steps - 1, i][0] ** 2 + x[steps - 1, i][1] ** 2

def forward():
    for s in range(steps - 1):
        func(s)
    compute_loss()
    

for i in range(N):
    x[0, i] = [1.0, 1.0]
loss.grad[None] = 1

with ti.ad.Tape(loss):
    forward()

print(loss[None])
print(x.grad[0, 99])