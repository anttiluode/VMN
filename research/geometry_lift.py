"""Exact complex-linear/conjugate decomposition of VMN's real tangent."""
import numpy as np


def real_to_complex(jacobian):
    j = np.asarray(jacobian)
    if (np.iscomplexobj(j) or j.ndim != 2 or j.shape[0] != j.shape[1]
        or j.shape[0] < 2 or j.shape[0] % 2 or not np.all(np.isfinite(j))):
        raise ValueError("Jacobian must be a finite real square matrix of even size")
    xx,xy,yx,yy = j[0::2,0::2],j[0::2,1::2],j[1::2,0::2],j[1::2,1::2]
    return (xx+yy+1j*(yx-xy))/2,(xx-yy+1j*(yx+xy))/2


def unitary_lift(n):
    if not isinstance(n,(int,np.integer)) or n < 1:
        raise ValueError("unit count must be a positive integer")
    t = np.zeros((2*n,2*n),complex)
    k = np.arange(n)
    t[k,2*k] = t[n+k,2*k] = 1/np.sqrt(2)
    t[k,2*k+1] = 1j/np.sqrt(2)
    t[n+k,2*k+1] = -1j/np.sqrt(2)
    return t


def doubled_matrix(a,b):
    a,b = np.asarray(a,dtype=complex),np.asarray(b,dtype=complex)
    if (a.ndim != 2 or a.shape != b.shape or a.shape[0] != a.shape[1]
        or not len(a) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))):
        raise ValueError("complex tangent blocks must be finite matching square matrices")
    return np.block([[a,b],[b.conj(),a.conj()]])


def vortex_complex_tangent(positions,gamma,delta=0.):
    p,g = np.asarray(positions),np.asarray(gamma)
    if (np.iscomplexobj(p) or np.iscomplexobj(g) or p.ndim != 2
        or p.shape[1] != 2 or not len(p) or g.shape != (len(p),)
        or not np.all(np.isfinite(p)) or not np.all(np.isfinite(g))
        or not np.isfinite(delta) or delta < 0):
        raise ValueError("provide finite planar positions, matching circulations and nonnegative smoothing")
    z = p[:,0]+1j*p[:,1]
    w = z[:,None]-z[None,:]
    s = np.abs(w)**2+delta**2
    np.fill_diagonal(s,np.inf)
    if np.any(s <= 0):
        raise ValueError("unsmoothed coincident vortex positions are singular")
    ordinary = 1j*g[None,:]*delta**2/(2*np.pi*s**2)
    conjugate = -1j*g[None,:]*w**2/(2*np.pi*s**2)
    a,b = -ordinary.copy(),-conjugate.copy()
    np.fill_diagonal(a,ordinary.sum(axis=1))
    np.fill_diagonal(b,conjugate.sum(axis=1))
    return a,b


def model_complex_tangent(model,state):
    q = np.asarray(state)
    if np.iscomplexobj(q) or q.size != model.dimension or not np.all(np.isfinite(q)):
        raise ValueError("provide one finite real state matching the model dimension")
    q = q.reshape(-1,2)
    z = q[:,0]+1j*q[:,1]
    av,bv = vortex_complex_tangent(model.anchors+model.geometry*q,
                                  model.gamma,model.smoothing)
    c = model.cubic*(1+1j*model.beta)
    a = np.diag(model.mu-model.decay+1j*model.omega-2*c*np.abs(z)**2)
    b = np.diag(-c*z**2)
    return a+model.coupling*av,b+model.coupling*bv
