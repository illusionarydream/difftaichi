import taichi as ti
ti.init()

N = 16

x = ti.field(dtype=ti.f32, shape=N, needs_grad=True)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def func():
    for i in x:
       loss[None] += x[i] ** 2

@ti.kernel
def grad_fill(data: ti.template()):
    for i in data:
        data.grad[i] = 0.0

@ti.ad.grad_replaced
def func_forward():
    func()

@ti.ad.grad_for(func_forward)
def func_backward():
    grad_fill(x)


x.fill(1.0)

loss.grad[None] = 1.0

with ti.ad.Tape(loss):
    func_forward()


print(x.grad)