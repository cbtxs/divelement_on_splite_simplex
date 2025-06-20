
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from hdivs_fe_space_new import HdivSFESpace
from fealpy.functionspace import LagrangeFESpace

from scipy.sparse import csr_matrix, coo_matrix, bmat
from scipy.sparse.linalg import spsolve

from linear_elastic_pde import LinearElasticPDE

from sympy import symbols, sin, cos, Matrix, lambdify

from fealpy.tools.show import showmultirate
from fealpy.tools.show import show_error_table

from scipy.sparse import csr_matrix
from mumps import DMumpsContext
from scipy.sparse.linalg import minres, gmres, lgmres

import sys
import time

def Solve(A, b):
    ctx = DMumpsContext()
    ctx.set_silent()
    ctx.set_centralized_sparse(A)

    x = bm.array(b)

    ctx.set_rhs(x)
    ctx.run(job=6)
    ctx.destroy()

    x = list(x)
    #x, _ = lgmres(A, b)
    return x
    

def plot_linear_function_u(uh, u):
    fig = plt.figure()
    fig.set_facecolor('white')
    axes = plt.axes(projection='3d')

    NC = mesh.number_of_cells()

    mid = mesh.entity_barycenter("cell")
    node = mesh.entity("node")
    cell = mesh.entity("cell")

    coor = node[cell]
    val = u(node).reshape(-1) 

    bcs = bm.eye(3, dtype=uh.dtype)
    uhval = uh(bcs) # (NC, 3)
    for ii in range(NC):
        axes.plot_trisurf(coor[ii, :, 0], coor[ii, :, 1], uhval[ii], color = 'r', lw=0.0)#数值解图像

    fig = plt.figure()
    fig.set_facecolor('white')
    axes = plt.axes(projection='3d')
    for ii in range(NC):
        axes.plot_trisurf(coor[ii, :, 0], coor[ii, :, 1], val[cell[ii]], color = 'b', lw=0.0)
        axes.plot_trisurf(coor[ii, :, 0], coor[ii, :, 1], uhval[ii], color = 'r', lw=0.0)#数值解图像
    plt.show()


def plot_linear_function(uh, u, component):
    fig = plt.figure()
    fig.set_facecolor('white')
    axes = plt.axes(projection='3d')


    NC = mesh.number_of_cells()

    mid = mesh.entity_barycenter("cell")
    node = mesh.entity("node")
    cell = mesh.entity("cell")
    cb   = mesh.entity_barycenter("cell")
    coor = node[cell]

    val = u(node).reshape(-1) 

    bcs = bm.eye(3, dtype=uh.dtype)
    uhval = uh(bcs) #(3, NC, NVC, 3)
    for ii in range(NC):
        for jj in range(3):
            vertex = node[cell[ii]]
            cbi = cb[ii]
            vertex[jj] = cbi
            axes.plot_trisurf(vertex[:, 0], vertex[:, 1], 
                              uhval[jj, ii, :, component], color = 'r', lw=0.0)

    fig = plt.figure()
    fig.set_facecolor('white')
    axes = plt.axes(projection='3d')
    for ii in range(NC):
        axes.plot_trisurf(coor[ii, :, 0], coor[ii, :, 1], 
                          val[cell[ii]], color = 'b', lw=0.0)
    for ii in range(NC):
        for jj in range(3):
            vertex = node[cell[ii]]
            cbi = cb[ii]
            vertex[jj] = cbi
            axes.plot_trisurf(vertex[:, 0], vertex[:, 1], 
                              uhval[jj, ii, :, component], color = 'r', lw=0.0)
    plt.show()


def mass_matrix(space : HdivSFESpace, lambda_0 : float, mu_0 : float):
    p = space.p
    mesh = space.mesh
    gdof = space.number_of_global_dofs()
    NC   = mesh.number_of_cells()

    cellmeasure = mesh.entity_measure('cell')[:, None]/3
    qf = mesh.quadrature_formula(p+3, 'cell')

    bcs, ws = qf.get_quadrature_points_and_weights()
    NQ = len(ws)
    phi = space.basis(bcs)
    trphi = phi[..., 0] + phi[..., -1]

    num = bm.array([1, 2, 1], dtype=phi.dtype)
    A = lambda0*bm.einsum('q, ct, tcqld, tcqmd, d->clm', ws, cellmeasure, phi, phi, num)
    A -= lambda1*bm.einsum('q, ct, tcql, tcqm->clm', ws, cellmeasure, trphi, trphi)

    cell2dof = space.cell_to_dof()
    I = bm.broadcast_to(cell2dof[:, None], A.shape)
    J = bm.broadcast_to(cell2dof[..., None], A.shape)
    A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof),
                   dtype=phi.dtype)
    return A

def mix_matrix(space0 : HdivSFESpace, space1 : LagrangeFESpace):
    p = space0.p
    mesh = space0.mesh
    NC   = mesh.number_of_cells()
    gdof0 = space0.number_of_global_dofs()
    gdof1 = space1.number_of_global_dofs()

    cell2dof0 = space0.cell_to_dof()
    cell2dof1 = space1.cell_to_dof()

    cellmeasure = mesh.entity_measure('cell')
    qf = mesh.quadrature_formula(p+3, 'cell')

    bcs, ws = qf.get_quadrature_points_and_weights()
    NQ = len(ws)
    phi = space0.div_basis(bcs)

    ldof = space1.number_of_local_dofs()
    psi = space1.basis(bcs)
    B_ = bm.einsum('q, c, cqld, cqm->clmd', ws, cellmeasure, phi, psi)

    shape = B_.shape[:-1]

    B = coo_matrix((gdof0, gdof1*2), dtype=phi.dtype)
    I = bm.broadcast_to(cell2dof0[..., None], shape)
    for i in range(2):
        J = bm.broadcast_to(gdof1*i + cell2dof1[:, None], shape)
        B += coo_matrix((B_[..., i].flat, (I.flat, J.flat)), 
                        shape=(gdof0, gdof1*2), dtype=phi.dtype)
    return B.tocsr()

def source_vector(space : LagrangeFESpace, f : callable):
    p = space.p
    mesh = space.mesh
    gdof = space.number_of_global_dofs()

    cellmeasure = mesh.entity_measure('cell')
    qf = mesh.quadrature_formula(p+3, 'cell')

    bcs, ws = qf.get_quadrature_points_and_weights()
    points = mesh.bc_to_point(bcs)

    phi  = space.basis(bcs)
    fval = f(points)
    b = bm.einsum('q, c, cql, cqd->cld', ws, cellmeasure, phi, fval)

    cell2dof = space.cell_to_dof()
    r = bm.zeros(gdof*2, dtype=phi.dtype)
    for i in range(2):
        bm.add.at(r, gdof*i + cell2dof, b[..., i]) 
    return r

def displacement_boundary_condition(space : HdivSFESpace, g : callable):
    p = space.p
    mesh = space.mesh
    TD = mesh.top_dimension()
    ldof = space.number_of_local_dofs()
    gdof = space.number_of_global_dofs()

    bdedge = mesh.boundary_edge_flag()
    e2c = mesh.edge_to_cell()[bdedge]
    en  = mesh.edge_unit_normal()[bdedge]
    cell2dof = space.cell_to_dof()[e2c[:, 0]]
    NBF = bdedge.sum()

    cellmeasure = mesh.entity_measure('edge')[bdedge]
    qf = mesh.quadrature_formula(p+3, 'edge')

    bcs, ws = qf.get_quadrature_points_and_weights()
    NQ = len(bcs)

    bcsi = [bm.insert(bcs, i, 0, axis=-1) for i in range(3)]

    symidx = [[0, 1], [1, 2]]
    phin = bm.zeros((NBF, NQ, ldof, 2), dtype=space.ftype)
    gval = bm.zeros((NBF, NQ, 2), dtype=space.ftype)
    for i in range(3):
        flag = e2c[:, 2] == i
        if flag.sum() == 0:
            continue

        phi = space.basis(bcsi[i])[i, e2c[flag, 0]] 
        phin[flag, ..., 0] = bm.sum(phi[..., symidx[0]] * en[flag, None, None], axis=-1)
        phin[flag, ..., 1] = bm.sum(phi[..., symidx[1]] * en[flag, None, None], axis=-1)
        points = mesh.bc_to_point(bcsi[i])[e2c[flag, 0]]
        gval[flag] = g(points)

    b = bm.einsum('q, c, cqld, cqd->cl', ws, cellmeasure, phin, gval)
    cell2dof = space.cell_to_dof()[e2c[:, 0]]
    r = bm.zeros(gdof, dtype=phi.dtype)
    bm.add.at(r, cell2dof, b) 
    return r

def right_hand_side(space : HdivSFESpace, f : callable):
    mesh = space.mesh
    p = space.p
    gdof = space.number_of_global_dofs()

    cellmeasure = mesh.entity_measure('cell')[..., None]/3
    qf = mesh.quadrature_formula(p+3, 'cell')
    bcs, ws = qf.get_quadrature_points_and_weights()

    dphi = space.div_basis(bcs) # (3, NC, NQ, ldof, 2)

    points = mesh.bc_to_point(bcs)
    fval = f(points) # (NC, NQ, 2)

    b_ = bm.einsum('q, ct, cqld, cqd->cl', ws, cellmeasure, dphi, fval)
    cell2dof = space.cell_to_dof()
    r = bm.zeros(gdof, dtype=dphi.dtype)
    bm.add.at(r, cell2dof, b_)
    return r

def stress_error(sigmah, sigma):
    space = sigmah.space
    mesh = space.mesh
    p = space.p

    cellmeasure = mesh.entity_measure('cell')[..., None]/3
    qf = mesh.quadrature_formula(p+3, 'cell')
    bcs, ws = qf.get_quadrature_points_and_weights()

    shval = sigmah(bcs) # (3, NC, NQ, 3)
    sval  = bm.zeros(shval.shape, dtype=shval.dtype)
    for i in range(3):
        bcsi = space.partition_bc_to_global_bc(bcs, i)
        points = mesh.bc_to_point(bcsi)
        sval[i] = sigma(points)

    num = bm.array([1, 2, 1], dtype=shval.dtype)
    err = shval - sval
    err = bm.einsum('q, ct, tcqd, d->c', ws, cellmeasure, err**2, num)
    return bm.sqrt(err.sum())

def div_matrix(space : HdivSFESpace):
    p = space.p
    mesh = space.mesh
    gdof = space.number_of_global_dofs()
    NC   = mesh.number_of_cells()

    cellmeasure = mesh.entity_measure('cell')
    qf = mesh.quadrature_formula(p+3, 'cell')

    bcs, ws = qf.get_quadrature_points_and_weights()
    NQ = len(ws)
    dphi = space.div_basis(bcs)
    A = bm.einsum('q, c, cqld, cqmd->clm', ws, cellmeasure, 
                          dphi, dphi)

    cell2dof = space.cell_to_dof()
    I = bm.broadcast_to(cell2dof[:, None], A.shape)
    J = bm.broadcast_to(cell2dof[..., None], A.shape)
    A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof),
                   dtype=dphi.dtype)
    return A


def solve(pde, N, p=1):
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=N, ny=N)
    space0 = HdivSFESpace(mesh, p=p)
    space1 = LagrangeFESpace(mesh, p=p-1, ctype='D')

    lambda0 = pde.lambda0
    lambda1 = pde.lambda1

    gdof0 = space0.number_of_global_dofs()
    gdof1 = space1.number_of_global_dofs()

    alpha = 0

    A = mass_matrix(space0, lambda0, lambda1)
    A += alpha*div_matrix(space0)
    B = mix_matrix(space0, space1)
    a = displacement_boundary_condition(space0, pde.displacement)
    b = source_vector(space1, pde.source)
    c = alpha*right_hand_side(space0, pde.source)

    A = bmat([[A, B], [B.T, None]], format='csr', dtype=A.dtype) 

    F = bm.zeros(A.shape[0], dtype=A.dtype)
    F[:gdof0] = a - c
    F[gdof0:] = -b

    X = Solve(A, F)

    sigmaval = X[:gdof0]
    u0val = X[gdof0:gdof0+gdof1]
    u1val = X[gdof0+gdof1:]

    sigmah = space0.function()
    sigmah[:] = sigmaval

    uh0 = space1.function()
    uh1 = space1.function()
    uh0[:] = u0val
    uh1[:] = u1val

    return sigmah, uh0, uh1


if __name__ == "__main__":
    lambda0 = 4
    lambda1 = 1
    maxit = 5
    p = int(sys.argv[1])

    errorType = []
    ps = [2, 3, 5]
    for p in ps:
        errorType += ['$|| \\boldsymbol{\\sigma} - \\boldsymbol{\\sigma}_h||_{0}$ for p = %i' % p, 
             '$|| \\boldsymbol{u} - \\boldsymbol{u}_h||_{0} $ for p = %i' % p]
    errorMatrix = bm.zeros((2*len(ps), maxit), dtype=bm.float64)
    h = bm.zeros(maxit, dtype=bm.float64)
    for j in range(len(ps)):
        for i in range(maxit):
            p = ps[j]
            N = 2**(i+1) 

            x, y = symbols('x y')
            pi = bm.pi 
            u0 = (sin(pi*x)*sin(pi*y))**2
            u1 = (sin(pi*x)*sin(pi*y))**2
            u0 = sin(5*x)*sin(7*y)
            u1 = cos(5*x)*cos(4*y)
            #u0 = x**2
            #u1 = y**2

            u = [u0, u1]
            pde = LinearElasticPDE(u, lambda0, lambda1)

            sigmah, uh0, uh1 = solve(pde, N, p=p)
            mesh = sigmah.space.mesh

            u0 = lambda p : pde.displacement(p)[..., 0]
            u1 = lambda p : pde.displacement(p)[..., 1]

            e0 = mesh.error(uh0, u0)
            e1 = mesh.error(uh1, u1)
            e2 = stress_error(sigmah, pde.stress)

            h[i] = 1/N
            errorMatrix[j*2+0, i] = e2
            errorMatrix[j*2+1, i] = bm.sqrt(e0**2 + e1**2)
            print(N, e0, e1, e2)

            stress0 = lambda p : pde.stress(p)[..., 0]
            stress1 = lambda p : pde.stress(p)[..., 1]
            stress2 = lambda p : pde.stress(p)[..., 2]

            #plot_linear_function(sigmah, stress2, 2)
            #plot_linear_function_u(uh0, u0)

    show_error_table(h, errorType, errorMatrix)
    showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
    plt.show()























