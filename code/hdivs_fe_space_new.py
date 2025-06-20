
from typing import Optional, TypeVar, Union, Generic, Callable
from fealpy.typing import TensorLike, Index, _S, Threshold

from fealpy.backend import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.mesh.mesh_base import Mesh
from fealpy.functionspace import FunctionSpace, BernsteinFESpace
from fealpy.functionspace.function import Function
from fealpy.functionspace.functional import symmetry_span_array, symmetry_index
from fealpy.decorator import barycentric, cartesian

import time

def number_of_multiindex(p, d):
    if d == 1:
        return p+1
    elif d == 2:
        return (p+1)*(p+2)//2
    elif d == 3:
        return (p+1)*(p+2)*(p+3)//6

def multiindex_to_number(a):
    d = a.shape[1] - 1
    if d==1:
        return a[:, 1]
    elif d==2:
        a1 = a[:, 1] + a[:, 2]
        a2 = a[:, 2]
        return a1*(1+a1)//2 + a2 
    elif d==3:
        a1 = a[:, 1] + a[:, 2] + a[:, 3]
        a2 = a[:, 2] + a[:, 3]
        a3 = a[:, 3]
        return a1*(1+a1)*(2+a1)//6 + a2*(1+a2)//2 + a3

class TensorDofsOnSubsimplex():
    def __init__(self, dofs : list, subsimplex : list, boundary = True):
        """
        dofs: list of tuple (alpha, I), alpha is the multi-index, I is the
              tensor index.
        """
        if isinstance(dofs, list):
            self.dof_scalar = bm.array([dof[0] for dof in dofs], dtype=bm.int32)
            self.dof_tensor = bm.array([dof[1] for dof in dofs], dtype=bm.int32)
        elif isinstance(dofs, tuple):
            self.dof_scalar = dofs[0]
            self.dof_tensor = dofs[1]

        self.subsimplex = subsimplex

        if boundary:
            self.dof2num = self._get_dof_to_num()

    def __getitem__(self, idx):
        return self.dof_scalar[idx], self.dof_tensor[idx]

    def __len__(self):
        return self.dof_scalar.shape[0]

    def _get_dof_to_num(self):
        alpha = self.dof_scalar
        I     = self.dof_tensor
        ldof  = number_of_multiindex(bm.sum(alpha[0]), alpha.shape[1]-1)
        idx = multiindex_to_number(alpha) + I*ldof

        nummap = bm.zeros((idx.max()+1,), dtype=alpha.dtype)
        nummap[idx] = bm.arange(len(idx), dtype=alpha.dtype)
        return nummap

    def permute_to_order(self, perm):
        alpha = self.dof_scalar.copy()
        alpha[:, self.subsimplex] = alpha[:, self.subsimplex][:, perm]

        I     = self.dof_tensor
        ldof  = number_of_multiindex(bm.sum(alpha[0]), alpha.shape[1]-1)
        idx = multiindex_to_number(alpha) + I*ldof
        return self.dof2num[idx]

    def dof_to_number(self, alpha, I):
        ldof  = number_of_multiindex(bm.sum(alpha), alpha.shape[0]-1)
        idx = multiindex_to_number(alpha) + I*ldof
        return self.dof2num[idx]


class HdivSFECellDof2d():
    def __init__(self, mesh : Mesh, p: int):
        self.p = p
        self.mesh = mesh
        self.TD = mesh.top_dimension() 

        self._get_simplex()
        self.boundary_dofs, self.internal_dofs = self.dof_classfication()

    def _get_simplex(self):
        TD = self.TD 
        mesh = self.mesh

        localnode = bm.array([[0], [1], [2]], dtype=mesh.itype)
        localEdge = bm.array([[1, 2], [0, 2], [0, 1]], dtype=mesh.itype)
        localcell = bm.array([[0, 1, 2]], dtype=mesh.itype)
        self.subsimplex = [localnode, localEdge, localcell]

        dual = lambda alpha : [i for i in range(self.TD+1) if i not in alpha]
        self.dual_subsimplex = [[dual(f) for f in ssixi] for ssixi in self.subsimplex]

    def dof_classfication(self):
        """
        Classify the dofs by the the entities.
        """
        p = self.p
        mesh = self.mesh
        TD = mesh.top_dimension()
        NS = TD*(TD+1)//2
        multiindex = bm.multi_index_matrix(self.p, TD)

        boundary_dofs = [[] for i in range(TD+1)]
        internal_dofs = [[] for i in range(TD+1)]

        # 边上的自由度
        fs = self.subsimplex[1] 
        fds = self.dual_subsimplex[1] 
        for j in range(len(fs)):
            fflag = bm.all(multiindex[:, fds[j]] == 0, axis=-1)
            N = fflag.sum()

            bscalar_ = bm.zeros((N*2, 3), dtype=mesh.itype)
            btensor_ = bm.zeros((N*2,  ), dtype=mesh.itype)
            iscalar_ = bm.zeros((N-2, 3), dtype=mesh.itype)
            itensor_ = bm.zeros((N-2,  ), dtype=mesh.itype)
            # 顶点上的自由度
            for v in range(2):
                fvflag = fflag & (multiindex[:, fs[j, v]] == self.p)
                bscalar_[v*2:v*2+2] = multiindex[fvflag][None, :]

            ifflag = fflag & bm.all(multiindex[:, fs[j]] != 0, axis=1)
            bscalar_[4::2] = multiindex[ifflag] 
            bscalar_[5::2] = multiindex[ifflag] 
            iscalar_[:] = multiindex[ifflag]

            btensor_[::2] = 0
            btensor_[1::2] = 1
            itensor_[:] = 2

            if N > 0:
                bfdofs = TensorDofsOnSubsimplex((bscalar_, btensor_), fs[j])
                boundary_dofs[1].append(bfdofs)
            if N > 2:
                ifdofs = TensorDofsOnSubsimplex((iscalar_, itensor_), fs[j], False)
                internal_dofs[1].append(ifdofs)

        # 单元上的自由度
        Tflag = bm.all(multiindex != 0, axis=-1)
        N = Tflag.sum()
        iscalar_ = bm.zeros((N*3, 3), dtype=mesh.itype)
        itensor_ = bm.zeros((N*3,  ), dtype=mesh.itype)

        for i in range(3):
            iscalar_[i::3] = multiindex[Tflag]
            itensor_[i::3] = i

        T = self.subsimplex[2][0]
        icdofs = TensorDofsOnSubsimplex((iscalar_, itensor_), T, False)
        internal_dofs[2].append(icdofs)
        return boundary_dofs, internal_dofs 

    def get_boundary_dof_from_dim(self, d):
        """
        Get the dofs of the entities of dimension d.
        """
        return self.boundary_dofs[d]

    def get_internal_dof_from_dim(self, d):
        """
        Get the dofs of the entities of dimension d.
        """
        return self.internal_dofs[d]

    def get_boundary_dof_from_subsimplex(self, ss):
        """
        Get the dofs of the subsimplex.
        """
        d = len(ss)-1
        for dof in self.boundary_dofs[d]:
            if dof.subsimplex == ss:
                return dof
        ValueError("The subsimplex is not found!")

class HdivSFEDof():
    """ 
    @brief: The class of HdivS finite element space dofs.
    @note: Only support the simplicial mesh, the order of  
            local edge, face of the mesh is the same as the order of subsimplex.
    """
    def __init__(self, mesh: Mesh, p: int):
        self.mesh = mesh
        self.p = p
        self.ftype = mesh.ftype
        self.itype = mesh.itype
        self.device = mesh.device

        self.cell_dofs = HdivSFECellDof2d(mesh, p)

    def number_of_local_dofs(self) -> int:
        """
        Get the number of local dofs on cell 
        """
        p = self.p
        return 3*(p+1)*(p+2)//2 + 3 

    def number_of_internal_local_dofs(self, doftype : str='cell') -> int:
        """
        Get the number of internal local dofs of the finite element space.
        """
        p = self.p
        TD = self.mesh.top_dimension()
        NS = TD*(TD+1)//2
        ldof = self.number_of_local_dofs()
        if doftype == 'cell':
            return ldof - 6*(p+1)  
        elif doftype == 'face' or doftype == 'edge':
            return 2*(p+1)
        elif doftype == 'node':
            return 0
        else:
            raise ValueError("Unknown doftype: {}".format(doftype))

    def number_of_global_dofs(self) -> int:
        """
        Get the number of global dofs of the finite element space.
        """
        mesh = self.mesh
        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()

        cldof = self.number_of_internal_local_dofs('cell')
        eldof = self.number_of_internal_local_dofs('edge')
        return NC*cldof + NE*eldof 

    def node_to_local_dof(self) -> TensorLike:
        """
        三个顶点上的自由度
        """
        p = self.p
        n2en = bm.array([[2*p+2, 2*p+3, 4*p+4, 4*p+5],
                         [0, 1, 4*p+6, 4*p+7],
                         [2, 3, 2*p+4, 2*p+5]], dtype=self.itype, device=self.device)
        return n2en

    def edge_to_internal_dof(self) -> TensorLike:
        """
        Get the index array of the dofs defined on the edges of the mesh.
        """
        mesh = self.mesh
        NE = mesh.number_of_edges()
        eldof = self.number_of_internal_local_dofs('edge')

        edge2dof = bm.arange(NE*eldof, dtype=self.itype, device=self.device)
        return edge2dof.reshape(NE, eldof)

    edge_to_dof = edge_to_internal_dof

    def cell_to_internal_dof(self) -> TensorLike:
        """
        Get the index array of the dofs defined on the cells of the mesh.
        """
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        eldof = self.number_of_internal_local_dofs('edge')
        cldof = self.number_of_internal_local_dofs('cell')

        N = NE*eldof
        cell2dof = bm.arange(N, N+NC*cldof, dtype=self.itype, device=self.device)
        return cell2dof.reshape(NC, cldof)

    def cell_to_dof(self, index: Index=_S) -> TensorLike:
        """
        Get the cell to dof map of the finite element space.
        """
        p = self.p
        mesh = self.mesh
        ldof = self.number_of_local_dofs()

        NC = mesh.number_of_cells()
        cell = mesh.entity('cell')
        edge = mesh.entity('edge')
        c2e  = mesh.cell_to_edge()

        edofs = self.cell_dofs.get_boundary_dof_from_dim(1)

        edge2idof = self.edge_to_internal_dof()
        cell2idof = self.cell_to_internal_dof()

        c2d = bm.zeros((NC, ldof), dtype=self.itype, device=self.device)
        idx = 0 # 统计自由度的个数

        # 边自由度
        inverse_perm = [1, 0]
        for e, dof in enumerate(edofs):
            n = len(dof)

            #le = bm.sort(mesh.localEdge[e])
            le = self.cell_dofs.subsimplex[1][e] 
            flag = cell[:, le[0]] != edge[c2e[:, e], 0]

            c2d[:, idx:idx+n] = edge2idof[c2e[:, e]]

            inverse_dofidx = dof.permute_to_order(inverse_perm)
            c2d[flag, idx:idx+n] = edge2idof[c2e[flag, e]][:, inverse_dofidx]
            idx += n

        # 单元自由度
        c2d[:, idx:] = cell2idof
        return c2d

    def is_boundary_dof(self, threshold=None, method=None) -> TensorLike:
        """
        Get the bool array of the boundary dofs.
        """
        pass

class HdivSFESpace(FunctionSpace):
    def __init__(self, mesh, p: int=1, ctype='C'):
        self.mesh = mesh
        self.p = p

        self.dof = HdivSFEDof(mesh, p)

        self.ftype = mesh.ftype
        self.itype = mesh.itype

        self.scalar_space = BernsteinFESpace(mesh, p)

        self.device = mesh.device
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

    def __str__(self):
        return "HdivSFESpace on {} with p={}".format(self.mesh, self.p)

    ## 自由度接口
    def number_of_local_dofs(self, doftype='cell') -> int:
        return self.dof.number_of_local_dofs()

    def number_of_global_dofs(self) -> int:
        return self.dof.number_of_global_dofs()

    def interpolation_points(self) -> TensorLike:
        return self.dof.interpolation_points()

    def cell_to_dof(self, index: Index=_S) -> TensorLike:
        return self.dof.cell_to_dof(index=index)

    def face_to_dof(self, index: Index=_S) -> TensorLike:
        return self.dof.face_to_dof(index=index)

    def edge_to_dof(self, index=_S):
        return self.dof.edge_to_dof(index=index)

    def is_boundary_dof(self, threshold=None, method=None) -> TensorLike:
        return self.dof.is_boundary_dof(threshold, method=method)

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    def project(self, u: Union[Callable[..., TensorLike], TensorLike],) -> TensorLike:
        pass

    def interpolate(self, u: Union[Callable[..., TensorLike], TensorLike],) -> TensorLike:
        pass

    def boundary_interpolate(self,
            gd: Union[Callable, int, float, TensorLike],
            uh: Optional[TensorLike] = None,
            *, threshold: Optional[Threshold]=None, method=None) -> TensorLike:
        #return self.function(uh), isDDof
        pass

    set_dirichlet_bc = boundary_interpolate

    def dof_frame(self) -> TensorLike:
        mesh = self.mesh

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        eframe = bm.zeros((NE, 2, 2), dtype=mesh.ftype)
        cframe = bm.zeros((NC, 2, 2), dtype=mesh.ftype)

        node = mesh.entity('node')
        edge = mesh.entity('edge')
        cell = mesh.entity('cell')
        c2e  = mesh.cell_to_edge()

        cframe[:, 0] = bm.array([[1, 0]], dtype=mesh.ftype)
        cframe[:, 1] = bm.array([[0, 1]], dtype=mesh.ftype)

        eframe[:, 0] = mesh.edge_unit_normal()
        eframe[:, 1] = mesh.edge_unit_tangent()

        edges = self.dof.cell_dofs.subsimplex[1]
        nframe = bm.zeros((NC, 3, 2, 2), dtype=mesh.ftype)
        for v in range(3):
            nframe[:, v, 0] = eframe[c2e[:, edges[v, 0]], 0]
            nframe[:, v, 1] = eframe[c2e[:, edges[v, 1]], 0]
        return nframe, eframe, cframe

    def basis_frame(self):
        mesh = self.mesh

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        eframe = bm.zeros((NE, 2, 2), dtype=mesh.ftype)
        cframe = bm.zeros((NC, 2, 2), dtype=mesh.ftype)
        c2e = mesh.cell_to_edge()

        node = mesh.entity('node')
        edge = mesh.entity('edge')
        cell = mesh.entity('cell')

        cframe[:, 0] = bm.array([[1, 0]], dtype=mesh.ftype)
        cframe[:, 1] = bm.array([[0, 1]], dtype=mesh.ftype)

        eframe[:, 0] = mesh.edge_unit_normal()
        eframe[:, 1] = mesh.edge_unit_tangent()

        edges = self.dof.cell_dofs.subsimplex[1]
        nframe = bm.zeros((NC, 3, 2, 2), dtype=mesh.ftype)
        for i in range(3):
            nframe[:, i, 0] = eframe[c2e[:, edges[i, 0]], 0]
            nframe[:, i, 1] = eframe[c2e[:, edges[i, 1]], 0]
        nframe = bm.linalg.inv(nframe)
        return nframe, eframe, cframe


    def dof_frame_of_S(self):
        mesh = self.mesh

        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        nframe, eframe, cframe = self.dof_frame()
        multiindex = bm.multi_index_matrix(2, 1)
        idx, num = symmetry_index(2, 2)

        esframe = bm.zeros((NE, 3, 3), dtype=self.ftype)
        for i, alpha in enumerate(multiindex): 
            esframe[:, i] = symmetry_span_array(eframe, alpha).reshape(NE, -1)[:, idx]

        csframe = bm.zeros((NC, 3, 3), dtype=self.ftype)
        for i, alpha in enumerate(multiindex): 
            csframe[:, i] = symmetry_span_array(cframe, alpha).reshape(NC, -1)[:, idx]

        
        n2e = self.dof.cell_dofs.subsimplex[1]
        nsframe = bm.zeros((NC, 3, 4, 3), dtype=self.ftype)
        c2e = mesh.cell_to_edge()
        for i in range(3):
            e = n2e[i]
            nsframe[:, i, :2] = esframe[c2e[:, n2e[i, 0]], :2]
            nsframe[:, i, 2:] = esframe[c2e[:, n2e[i, 1]], :2]
        return nsframe, esframe, csframe

    def basis_frame_of_S(self):
        p = self.p
        mesh = self.mesh
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        edge = mesh.entity('edge')

        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        nframe, eframe, cframe = self.basis_frame() #nframe: (NC, 3, 2, 2)
        multiindex = bm.multi_index_matrix(2, 1)
        idx, num = symmetry_index(2, 2)

        esframe = bm.zeros((NE, 3, 3), dtype=self.ftype)
        for i, alpha in enumerate(multiindex): 
            esframe[:, i] = symmetry_span_array(eframe, alpha).reshape(NE, -1)[:, idx]

        csframe = bm.zeros((NC, 3, 3), dtype=self.ftype)
        for i, alpha in enumerate(multiindex): 
            csframe[:, i] = symmetry_span_array(cframe, alpha).reshape(NC, -1)[:, idx]

        cb = mesh.bc_to_point(bm.array([[1/3, 1/3, 1/3]], dtype=self.ftype))[:, 0]
        n2e = self.dof.cell_dofs.subsimplex[1]
        nsframe = bm.zeros((3, NC, 3, 5, 3), dtype=self.ftype)

        nframe = nframe.reshape(-1, 2, 2)
        for i, alpha in enumerate(multiindex):
            val = symmetry_span_array(nframe, alpha).reshape(1, NC, 3, -1)
            nsframe[..., i, :] = val[..., idx] 

        R = bm.array([[0, -1], [1, 0]], dtype=self.ftype)
        glambda = self.grad_lambda_on_refine_mesh()@R
        glambda = glambda.reshape(3, NC, 3, 1, 2)
        for v in range(3):
            j, k = n2e[v]

            glambdavj = bm.concatenate([glambda[k, :, v], glambda[k, :, j]], axis=1) 
            glambdavj = symmetry_span_array(glambdavj, [1, 1]).reshape(NC, 4)[:, idx]
            glambdavk = bm.concatenate([glambda[j, :, v], glambda[j, :, k]], axis=1)
            glambdavk = symmetry_span_array(glambdavk, [1, 1]).reshape(NC, 4)[:, idx]

            nsframe[j, :, v, 3] =  2*(p+1)*glambdavk 
            nsframe[k, :, v, 3] = -2*(p+1)*glambdavj 

            glambdavv = bm.concatenate([glambda[j, :, v], glambda[j, :, v]], axis=1)
            glambdavv = symmetry_span_array(glambdavv, [1, 1]).reshape(NC, 4)[:, idx]
            nsframe[j, :, v, 4] = p*(p+1)*glambdavv

            glambdavv = bm.concatenate([glambda[k, :, v], glambda[k, :, v]], axis=1)
            glambdavv = symmetry_span_array(glambdavv, [1, 1]).reshape(NC, 4)[:, idx]
            nsframe[k, :, v, 4] = -p*(p+1)*glambdavv

        ndframe, _, _ = self.dof_frame_of_S() # (NC, 3, 4, 3)
        coeff = bm.zeros((NC, 3, 4, 4), dtype=self.ftype)
        for v in range(3):
            j, k = n2e[v]
            vdframe = ndframe[:, v] # (NC, 4, 3)
            vnsframej = nsframe[j, :, v, :4] # (NC, 4, 3)
            vnsframek = nsframe[k, :, v, :4]

            coeff[:, v, 0, :] = bm.einsum("cd, cbd, d-> cb", vdframe[:, 0], 
                                          vnsframej, num)
            coeff[:, v, 1, :] = bm.einsum("cd, cbd, d-> cb", vdframe[:, 1], 
                                          vnsframej, num)
            coeff[:, v, 2, :] = bm.einsum("cd, cbd, d-> cb", vdframe[:, 2],
                                          vnsframek, num)
            coeff[:, v, 3, :] = bm.einsum("cd, cbd, d-> cb", vdframe[:, 3],
                                          vnsframek, num)
        coeff = bm.linalg.inv(coeff)
        return nsframe, esframe, csframe, coeff

    def partition_bc_to_global_bc(self, bc: TensorLike, t : int) -> TensorLike:
        bcst = bc.copy()
        a = bc[:, t, None]/3
        bcst[:, t] = 0
        bcst += a
        return bcst

    def grad_lambda_on_refine_mesh(self):
        """
        """
        mesh = self.mesh
        cell = mesh.entity('cell')
        node = mesh.entity('node')
        cb   = mesh.entity_barycenter('cell')

        NC = mesh.number_of_cells()
        glambda = bm.zeros((3, NC, 3, 2), dtype=self.ftype)
        n2e = self.dof.cell_dofs.subsimplex[1]

        for v in range(3):
            j, k = n2e[v]
            tcj = node[cell[:, j]] - cb
            tck = node[cell[:, k]] - cb
            tcv = node[cell[:, v]] - cb

            R   = bm.array([[0, -1], [1, 0]], dtype=self.ftype)
            ncj = tcj@R
            nck = tck@R
            glambda[j, :, v] = nck/(bm.sum(nck*tcv, axis=1))[:, None]
            glambda[k, :, v] = ncj/(bm.sum(ncj*tcv, axis=1))[:, None]

        return glambda

    def basis(self, bcs: TensorLike, index: Index=_S) -> TensorLike:
        p = self.p
        mesh = self.mesh
        dof = self.dof

        ldof = dof.number_of_local_dofs()

        ndofs = dof.cell_dofs.get_boundary_dof_from_dim(0)
        edofs = dof.cell_dofs.get_boundary_dof_from_dim(1)

        iedofs = dof.cell_dofs.get_internal_dof_from_dim(1)
        icdofs = dof.cell_dofs.get_internal_dof_from_dim(2)

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        cell = mesh.entity('cell')
        c2e = mesh.cell_to_edge()
        bcs_part = bm.zeros((3, )+bcs.shape, dtype=self.ftype)

        nsframe, esframe, csframe, coeff = self.basis_frame_of_S()

        glambda = self.grad_lambda_on_refine_mesh()

        phi_s = self.scalar_space.basis(bcs, p=self.p)# (NC, NQ, ldof)
        phi_part = bm.zeros((3, )+phi_s.shape, dtype=self.ftype)
        for i in range(3):
            bcsi = self.partition_bc_to_global_bc(bcs, i)
            bcs_part[i] = bcsi
            phi_part[i] = self.scalar_space.basis(bcsi, p=self.p)

        NQ = bcs.shape[0]
        phi = bm.zeros((3, NC, NQ, ldof, 3), dtype=self.ftype)
        phiv = bm.zeros((3, NC, NQ, 3, 4, 3), dtype=self.ftype)

        n2e = self.dof.cell_dofs.subsimplex[1]
        n2ld = self.dof.node_to_local_dof()
        basisidx = [0, -p-1, -1]    
        
        for v in range(3):
            # 计算 phi[:, :, :, v, :3]
            j, k = n2e[v]
            tensor_part = nsframe[0, :, v, :3]     # (NC, 3, 3)
            scalar_part = phi_part[j, :, :, basisidx[v]] # (NC, NQ)
            phiv[j, :, :, v, :3] = bm.einsum("cq, cli-> cqli", scalar_part,
                                             tensor_part) 
            scalar_part = phi_part[k, :, :, basisidx[v]] # (NC, NQ)
            phiv[k, :, :, v, :3] = bm.einsum("cq, cli-> cqli", scalar_part,
                                             tensor_part)
            scalar_part = phi_part[v, :, :, basisidx[v]] # (NC, NQ)
            phiv[v, :, :, v, :3] = bm.einsum("cq, cli-> cqli", scalar_part,
                                             tensor_part)

            scalar_partv = bcs[None, :, v]**p#phi_s[:, :, basisidx[v]] # (NC, NQ)
            scalar_partk = bcs[None, :, v]**(p-1)*bcs[None, :, j]
            scalar_partj = bcs[None, :, v]**(p-1)*bcs[None, :, k]

            tensor_partv = nsframe[:, :, v, 4, None]      # (3, NC, NQ, 3)
            tensor_partk = nsframe[k, :, v, 3, None]
            tensor_partj = nsframe[j, :, v, 3, None]

            phiv[j, :, :, v, 3] = scalar_partj[..., None]*tensor_partv[j]
            phiv[j, :, :, v, 3] += scalar_partv[..., None]*tensor_partj
            phiv[k, :, :, v, 3] = scalar_partk[..., None]*tensor_partv[k]
            phiv[k, :, :, v, 3] += scalar_partv[..., None]*tensor_partk

            scalar_partk = bcs_part[:, None, :, v]**(p-1)*bcs_part[:, None, :, j]
            scalar_partj = bcs_part[:, None, :, v]**(p-1)*bcs_part[:, None, :, k]

            phiv[:, :, :, v, 3] -= scalar_partj[..., None]*tensor_partv[j]
            phiv[:, :, :, v, 3] -= scalar_partk[..., None]*tensor_partv[k]

            phiv[:, :, :, v] = bm.einsum("cij, tcqid-> tcqjd", 
                                         coeff[:, v], phiv[:, :, :, v])
            n2ldv = n2ld[v]
            phi[..., n2ldv, :] = phiv[..., v, :, :]

        # 边基函数
        idx = 0
        for e, edof in enumerate(edofs):
            N = len(edof)

            scalar_phi_idx = multiindex_to_number(edof.dof_scalar[4:])
            scalar_part = phi_part[..., scalar_phi_idx, None] # (3, NC, NQ, N, 1)
            tensor_part = esframe[c2e[:, e]][None, :, None, edof.dof_tensor[4:], :] # (1, NC, 1, N, 3)
            phi[..., idx+4:idx+N, :] =scalar_part * tensor_part 
            idx += N

        for e, edof in enumerate(iedofs):
            N = len(edof)
            scalar_phi_idx = multiindex_to_number(edof.dof_scalar)
            scalar_part = phi_part[..., scalar_phi_idx, None] # (3, NC, NQ, N, 1)
            tensor_part = esframe[c2e[:, e]][None, :, None, edof.dof_tensor, :]
            phi[..., idx:idx+N, :] = scalar_part * tensor_part
            idx += N

        # 单元基函数
        scalar_phi_idx = multiindex_to_number(icdofs[0].dof_scalar)
        scalar_part = phi_part[..., scalar_phi_idx, None] # (3, NC, NQ, N, 1)
        tensor_part = csframe[None, :, None, icdofs[0].dof_tensor, :] # (1, NC, 1, N, 3)
        phi[..., idx:, :] = scalar_part * tensor_part
        return phi

    def grad_vertex_basis(self, bcs: TensorLike, index: Index=_S) -> TensorLike:
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        NQ = bcs.shape[0]

        R = bm.array([[0, -1], [1, 0]], dtype=self.ftype)
        glambda = self.mesh.grad_lambda()
        gphi = bm.zeros((2, NC, NQ, 3, 2), dtype=self.ftype)
        nstar = self.dof.cell_dofs.dual_subsimplex[0]
        for v in range(3):
            j, k = nstar[v]
            scalarj = (p-1)*bcs[:, v]**(p-2)*bcs[:, j] # (NQ)
            scalark = (p-1)*bcs[:, v]**(p-2)*bcs[:, k]
            scalarv = bcs[:, v]**(p-1)
            gphi[0, :, :, v] = bm.einsum("q, cd-> cqd", scalarj, glambda[:, v])
            gphi[0, :, :, v] += bm.einsum("q, cd-> cqd", scalarv, glambda[:, j])
            gphi[1, :, :, v] = bm.einsum("q, cd-> cqd", scalark, glambda[:, v])
            gphi[1, :, :, v] += bm.einsum("q, cd-> cqd", scalarv, glambda[:, k])
        return gphi
            
    def div_basis(self, bcs: TensorLike): 
        p = self.p
        mesh = self.mesh
        dof = self.dof

        ldof = dof.number_of_local_dofs()

        ndofs = dof.cell_dofs.get_boundary_dof_from_dim(0)
        edofs = dof.cell_dofs.get_boundary_dof_from_dim(1)

        iedofs = dof.cell_dofs.get_internal_dof_from_dim(1)
        icdofs = dof.cell_dofs.get_internal_dof_from_dim(2)

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        cell = mesh.entity('cell')
        c2e = mesh.cell_to_edge()

        nsframe, esframe, csframe, coeff = self.basis_frame_of_S()
        gphi = self.scalar_space.grad_basis(bcs, p=self.p) # (NC, NQ, ldof)
        glambda = self.mesh.grad_lambda()
        gphiv = self.grad_vertex_basis(bcs) # (2, NC, NQ, 3, 2)
        NQ = bcs.shape[0]
        dphi = bm.zeros((NC, NQ, ldof, 2), dtype=self.ftype)
        dphiv = bm.zeros((NC, NQ, 3, 4, 2), dtype=self.ftype)

        nstar = self.dof.cell_dofs.dual_subsimplex[0]
        n2ld = self.dof.node_to_local_dof()
        basisidx = [0, -p-1, -1]    
        for v in range(3):
            # 计算 phi[:, :, :, v, :3]
            j, k = nstar[v]
            tensor_part = nsframe[0, :, v, :3]     # (NC, 3, 3)
            scalar_part = gphi[:, :, basisidx[v]] # (NC, NQ, 2)
            dphiv[:, :, v, :3, 0] = bm.einsum("cqd, cid-> cqi", scalar_part,
                                                tensor_part[..., :2]) 
            dphiv[:, :, v, :3, 1] = bm.einsum("cqd, cid-> cqi", scalar_part,
                                                tensor_part[..., 1:])

            gphivj = -gphiv[0, :, :, v] # (NC, NQ, 2)
            gphivk = -gphiv[1, :, :, v]

            tensor_partj = nsframe[k, :, v, 4, None]
            tensor_partk = nsframe[j, :, v, 4, None] # (NC, 1, 3)

            dphiv[:, :, v, 3, 0]  = bm.sum(gphivj*tensor_partj[..., :2], axis=-1)
            dphiv[:, :, v, 3, 0] += bm.sum(gphivk*tensor_partk[..., :2], axis=-1)
            dphiv[:, :, v, 3, 1]  = bm.sum(gphivj*tensor_partj[..., 1:], axis=-1)
            dphiv[:, :, v, 3, 1] += bm.sum(gphivk*tensor_partk[..., 1:], axis=-1)

            dphiv[:, :, v] = bm.einsum("cij, cqid-> cqjd", 
                                         coeff[:, v], dphiv[:, :, v])
            n2ldv = n2ld[v]
            dphi[..., n2ldv, :] = dphiv[..., v, :, :]

        # 边基函数
        idx = 0
        for e, edof in enumerate(edofs):
            N = len(edof)

            scalar_phi_idx = multiindex_to_number(edof.dof_scalar[4:])
            scalar_part = gphi[..., scalar_phi_idx, :] # (NC, NQ, N, 2)
            tensor_part = esframe[c2e[:, e]][:, None, edof.dof_tensor[4:], :] # (NC, 1, N, 3)
            dphi[..., idx+4:idx+N, 0] = bm.einsum("cqld, cqld-> cql",
                                         scalar_part, tensor_part[..., :2])
            dphi[..., idx+4:idx+N, 1] = bm.einsum("cqld, cqld-> cql",
                                         scalar_part, tensor_part[..., 1:])
            idx += N

        for e, edof in enumerate(iedofs):
            N = len(edof)
            scalar_phi_idx = multiindex_to_number(edof.dof_scalar)
            scalar_part = gphi[..., scalar_phi_idx, :] # (NC, NQ, N, 1)
            tensor_part = esframe[c2e[:, e]][:, None, edof.dof_tensor, :]
            dphi[..., idx:idx+N, 0] = bm.einsum("cqld, cqld-> cql",
                                           scalar_part, tensor_part[..., :2])
            dphi[..., idx:idx+N, 1] = bm.einsum("cqld, cqld-> cql",
                                        scalar_part, tensor_part[..., 1:])
            idx += N

        # 单元基函数
        scalar_phi_idx = multiindex_to_number(icdofs[0].dof_scalar)
        scalar_part = gphi[..., scalar_phi_idx, :] # (NC, NQ, N, 2)
        tensor_part = csframe[:, None, icdofs[0].dof_tensor, :] # (NC, 1, N, 3)
        dphi[..., idx:, 0] = bm.einsum("cqld, cqld-> cql",
                                      scalar_part, tensor_part[..., :2])
        dphi[..., idx:, 1] = bm.einsum("cqld, cqld-> cql",
                                      scalar_part, tensor_part[..., 1:])
        return dphi


    @barycentric
    def value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike: 
        if isinstance(bc, tuple):
            TD = len(bc)
        else :
            TD = bc.shape[-1] - 1
        phi = self.basis(bc, index=index)
        e2dof = self.dof.cell_to_dof()
        val = bm.einsum('tcqld, cl -> tcqd', phi, uh[e2dof])
        return val

    @barycentric
    def div_value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike:
        if isinstance(bc, tuple):
            TD = len(bc)
        else :
            TD = bc.shape[-1] - 1
        dphi = self.div_basis(bc)
        e2dof = self.dof.cell_to_dof()
        val = bm.einsum('cqld, cl -> cqd', dphi, uh[e2dof])
        return val
    
