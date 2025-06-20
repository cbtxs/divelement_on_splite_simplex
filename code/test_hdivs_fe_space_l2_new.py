
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh 
from hdivs_fe_space_new import HdivSFESpace 

import time

from fealpy.functionspace import LagrangeFESpace

from scipy.sparse import csr_matrix, coo_matrix, bmat
from scipy.sparse.linalg import spsolve

from linear_elastic_pde import LinearElasticPDE

from sympy import symbols, sin, cos, Matrix, lambdify

from fealpy.tools.show import showmultirate
from fealpy.tools.show import show_error_table

import sys


def mass_matrix(space : HdivSFESpace):
    p = space.p
    mesh = space.mesh
    gdof = space.number_of_global_dofs()
    NC   = mesh.number_of_cells()

    cellmeasure = mesh.entity_measure('cell')[:, None]/3
    qf = mesh.quadrature_formula(p+2, 'cell')

    bcs, ws = qf.get_quadrature_points_and_weights()
    NQ = len(ws)
    phi = space.basis(bcs)
    print(phi.shape)

    num = bm.array([1, 2, 1], dtype=phi.dtype)
    A = bm.einsum('q, ct, tcqld, tcqmd, d->clm', ws, cellmeasure, phi, phi, num)
    cell2dof = space.cell_to_dof()
    I = bm.broadcast_to(cell2dof[:, None], A.shape)
    J = bm.broadcast_to(cell2dof[..., None], A.shape)
    A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof),
                   dtype=phi.dtype)
    return A

def source_vector(space : HdivSFESpace, f):
    p = space.p
    mesh = space.mesh
    gdof = space.number_of_global_dofs()
    NC   = mesh.number_of_cells()

    cellmeasure = mesh.entity_measure('cell')[:, None]/3
    qf = mesh.quadrature_formula(p+2, 'cell')

    bcs, ws = qf.get_quadrature_points_and_weights()
    NQ = len(ws)
    phi = space.basis(bcs)
    fval = bm.zeros((3, NC, NQ, 3), dtype=phi.dtype)
    for i in range(3):
        bcsi = space.partition_bc_to_global_bc(bcs, i)
        point = mesh.bc_to_point(bcsi)
        fval[i] = f(point)

    num = bm.array([1, 2, 1], dtype=phi.dtype) 
    b_ = bm.einsum('q, ct, tcqld, tcqd, d->cl', ws, cellmeasure, phi, fval, num)
    cell2dof = space.cell_to_dof()
    b = bm.zeros(gdof, dtype=phi.dtype)
    bm.add.at(b, cell2dof, b_)
    return b

def cross(a, b):
    return a[0]*b[1] - a[1]*b[0]

def point_to_bc(vertices, point):
    v0 = vertices[0]
    v1 = vertices[1]
    v2 = vertices[2]
    area = cross(v1 - v0, v2 - v0)
    area0 = cross(v1 - point, v2 - point)
    area1 = cross(v2 - point, v0 - point)
    area2 = cross(v0 - point, v1 - point)
    bc = bm.array([area0/area, area1/area, area2/area], dtype=point.dtype)
    return bc

p = int(sys.argv[1])
node = bm.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=bm.float64)
cell = bm.array([[1, 2, 0], [3, 0, 2]], dtype=bm.int32)
#node = bm.array([[0, 0], [1, 0], [0, 1]], dtype=bm.float64)
#cell = bm.array([[0, 1, 2]], dtype=bm.int32)
#mesh = TriangleMesh(node, cell)
n=4
mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=n, ny=n)
space = HdivSFESpace(mesh, p=p)

lambda0 = 4
lambda1 = 1
maxit = 4
p = 1
N = 1 

x, y, z = symbols('x y z')

pi = bm.pi
u0 = x**2*y**2
u1 = y**2*z**2

u0 = x**2
u1 = x**2

u = [u0, u1]
pde = LinearElasticPDE(u, lambda0, lambda1)

u0 = lambda p : pde.displacement(p)[..., 0]
u1 = lambda p : pde.displacement(p)[..., 1]
stress0 = lambda p : pde.stress(p)[..., 0]
stress1 = lambda p : pde.stress(p)[..., 1]
stress2 = lambda p : pde.stress(p)[..., 2]

M = mass_matrix(space)
b = source_vector(space, pde.stress)
sigmah = space.function()
sigmah[:] = spsolve(M, b)

qf = mesh.quadrature_formula(3, 'cell')
bcs, ws = qf.get_quadrature_points_and_weights()

sigmah_val = sigmah(bcs)

NC = mesh.number_of_cells()
NQ = len(ws)
point = mesh.bc_to_point(bcs)

sigma_val0 = stress0(point)
sigma_val1 = stress1(point)
sigma_val2 = stress2(point)

diff = bm.abs(sigmah_val[..., 0] - sigma_val0)
print(bm.max(diff))
diff = bm.abs(sigmah_val[..., 1] - sigma_val1)
print(bm.max(diff))
diff = bm.abs(sigmah_val[..., 2] - sigma_val2)
print(bm.max(diff))

div_sigmah_val = sigmah.div_value(bcs)
div_sigma_val = -pde.source(point)
print(div_sigma_val[0, 0, 0])
print(div_sigmah_val[0, 0, 0])
diff = bm.abs(div_sigmah_val - div_sigma_val)
print("aaa : ", bm.max(diff))

if 1:
    def point_to_bc_(vertices, point):
        v0 = vertices[0]
        v1 = vertices[1]
        v2 = vertices[2]
        area = cross(v1 - v0, v2 - v0) 

        area0 = cross(v1 - point, v2 - point)
        area1 = cross(v2 - point, v0 - point)
        area2 = cross(v0 - point, v1 - point)
        bc = bm.array([area0/area, area1/area, area2/area], dtype=point.dtype)
        return bc

    def point_to_splite_bc(vertices, point):
        vc = bm.sum(vertices, axis=0)/3
        vertices_ = bm.zeros((3, 2), dtype=vertices.dtype)
        for i in range(3):
            vertices_[:] = vertices
            vertices_[i] = vc
            bc = point_to_bc_(vertices_, point)
            if bm.all(bc >= 0):
                return i, bc
        return -1, None

    def compute_sigmah_on_point(sigh, point):
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        cidx = 0
        lcidx = 0
        bc = 0
        for c in range(NC):
            vertice = node[cell[c]]
            lcidx, bc = point_to_splite_bc(vertice, point)
            if lcidx >= 0:
                cidx = c
                break
        val = sigh(bc[None, :])[lcidx, cidx, 0]
        return val

    def compute_div_sigmah_on_point(sigh, point):
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        cidx = 0
        lcidx = 0
        bc = 0
        for c in range(NC):
            vertice = node[cell[c]]
            bc = point_to_bc_(vertice, point)
            if bm.all(bc >= 0):
                cidx = c
                break
        val = sigh.div_value(bc[None, :])[cidx, 0]
        return val

    gdof = space.number_of_global_dofs()
    sigh = space.function()
    sigh[:] = 0
    point = bm.array([0.12, 0.03], dtype=bm.float64)
    delta = 1e-8 
    pointx = point + delta*bm.array([1, 0], dtype=bm.float64)
    pointy = point + delta*bm.array([0, 1], dtype=bm.float64)
    for i in range(gdof)[:]:
        sigh[i] = 1
        val  = compute_sigmah_on_point(sigh, point) 
        valx = compute_sigmah_on_point(sigh, pointx)
        valy = compute_sigmah_on_point(sigh, pointy)

        dif_sigh_x = (valx[[0, 1]] - val[[0, 1]])/delta
        dif_sigh_y = (valy[[1, 2]] - val[[1, 2]])/delta

        div_sigh = dif_sigh_x + dif_sigh_y
        div_sigmah_val = compute_div_sigmah_on_point(sigh, point) 
        diff = bm.abs(div_sigmah_val - div_sigh)
        if bm.max(diff) > 1e-6:
            print('error')
            print("dof : ", i)
            print("div_sigh", div_sigh)
            print("div_sigmah_val", div_sigmah_val)
            print("diff", diff)
        sigh[i] = 0























