#!/usr/bin/env python3
import argparse
from html import parser
import numpy as np
import h5py
from scipy.io import FortranFile

'''
module purge
module load intel/2023.2.0 impi/2021.10.0 hdf5/1.14.1-2 mkl/2023.2.0 python/3.12.1
  
python3 cfl3d_to_sod2d_source.py \
  --grid plot3dg_35.bin \
  --q plot3dq_35.bin \
  --z-planes 12 \
  --z-span 0.2 \
  --porder 2 \
  --mach 0.15 \
  --real float32 \
  --out-src-mesh cfl3d_z12.h5 \
  --out-src-restart restart_cfl3d_z12.h5 \
  --rho-constant-one
  
CFL3D Plot3D grid + solution into a source mesh/restart HDF5 pair 
suitable for SOD2D incompressible solver.

CFL3D plot3dg.bin + plot3dq.bin
        ↓
read grid coordinates and conservative variables
        ↓
convert CFL3D variables to SOD2D-style variables
        ↓
extrude/replicate 2D CFL3D data onto target SOD2D z-planes
        ↓
write:
  1) cfl3d_source_*.h5        → source coordinates
  2) restart_cfl3d_source_*.h5 → source flow fields
  
  
The CFL3D plot3d.bin files are expected to have 2 spanwise planes (2.5D), which are averaged together.
The SOD2D target .hdf mesh is used to determine the z planes to extract from the CFL3D data, 
but the x/y coordinates are taken from the CFL3D grid  and written to the output source mesh .hdf. 
The output restart .hdf contains the velocity and pressure fields for the SOD2D solver, converted from CFL3D's density/momentum/energy format.      


'''

#### reads nblocks ni, nj, nk x, y, z
def read_plot3d_grid_bin(fname, real_dtype=np.float64):
    f = FortranFile(fname, "r")

    nblocks = f.read_ints(np.int32)[0]
    dims_raw = f.read_ints(np.int32)

    if dims_raw.size == nblocks * 3:
        dims = dims_raw.reshape((nblocks, 3))
        is_2d = False
    elif dims_raw.size == nblocks * 2:
        dims2 = dims_raw.reshape((nblocks, 2))
        dims = np.column_stack([dims2, np.ones(nblocks, dtype=np.int32)])
        is_2d = True
    else:
        raise RuntimeError(f"Cannot interpret grid dimensions: nblocks={nblocks}, dims={dims_raw}")

    blocks = []

    for b in range(nblocks):
        ni, nj, nk = dims[b]
        n = ni * nj * nk

        data = f.read_reals(real_dtype)

        if is_2d:
            if data.size == 2 * n:
                x = data[0:n].reshape((ni, nj, nk), order="F")
                y = data[n:2*n].reshape((ni, nj, nk), order="F")
                z = np.zeros_like(x)

            elif data.size == 3 * n:
                x = data[0:n].reshape((ni, nj, nk), order="F")
                y = data[n:2*n].reshape((ni, nj, nk), order="F")
                z = data[2*n:3*n].reshape((ni, nj, nk), order="F")

            else:
                raise RuntimeError(
                    f"2D grid block {b}: expected {2*n} or {3*n}, got {data.size}"
                )

        blocks.append((x, y, z))

    f.close()
    return blocks

## The Q file contains conservative compressible variables:
## rho, rho*u, rho*v , rho*w, rho*E
def read_plot3d_q_bin(fname, real_dtype=np.float64):
    f = FortranFile(fname, "r")

    nblocks = f.read_ints(np.int32)[0]
    dims_raw = f.read_ints(np.int32)

    if dims_raw.size == nblocks * 3:
        dims = dims_raw.reshape((nblocks, 3))
        is_2d = False
    elif dims_raw.size == nblocks * 2:
        dims2 = dims_raw.reshape((nblocks, 2))
        dims = np.column_stack([dims2, np.ones(nblocks, dtype=np.int32)])
        is_2d = True
    else:
        raise RuntimeError(f"Cannot interpret Q dimensions: nblocks={nblocks}, dims={dims_raw}")

    blocks = []

    for b in range(nblocks):
        ni, nj, nk = dims[b]
        n = ni * nj * nk

        aux = f.read_reals(real_dtype)
        data = f.read_reals(real_dtype)

        if is_2d:
            if data.size == 4 * n:
                rho = data[0:n].reshape((ni, nj, nk), order="F")
                rhou = data[n:2*n].reshape((ni, nj, nk), order="F")
                rhov = data[2*n:3*n].reshape((ni, nj, nk), order="F")
                rhow = np.zeros_like(rho)
                rhoE = data[3*n:4*n].reshape((ni, nj, nk), order="F")
            elif data.size == 5 * n:
                rho = data[0:n].reshape((ni, nj, nk), order="F")
                rhou = data[n:2*n].reshape((ni, nj, nk), order="F")
                rhov = data[2*n:3*n].reshape((ni, nj, nk), order="F")
                rhow = data[3*n:4*n].reshape((ni, nj, nk), order="F")
                rhoE = data[4*n:5*n].reshape((ni, nj, nk), order="F")
            else:
                raise RuntimeError(f"2D Q block {b}: expected {4*n} or {5*n}, got {data.size}")
        else:
            if data.size != 5 * n:
                raise RuntimeError(f"3D Q block {b}: expected {5*n}, got {data.size}")

            rho = data[0:n].reshape((ni, nj, nk), order="F")
            rhou = data[n:2*n].reshape((ni, nj, nk), order="F")
            rhov = data[2*n:3*n].reshape((ni, nj, nk), order="F")
            rhow = data[3*n:4*n].reshape((ni, nj, nk), order="F")
            rhoE = data[4*n:5*n].reshape((ni, nj, nk), order="F")

        blocks.append((rho, rhou, rhov, rhow, rhoE))

    f.close()
    return blocks

def generate_z_planes(n_linear_planes, zspan, porder):
    n_target_planes = (n_linear_planes - 1) * porder + 1
    return np.linspace(0.0, zspan, n_target_planes)

# ## This opens the SOD2D target mesh:
# def get_target_z_planes(target_hdf):
#     with h5py.File(target_hdf, "r") as h5:
#         pts = h5["/VTKHDF/Points"][()]
#     z = pts[:, 2]
#     z_unique = np.unique(np.round(z, 8))
#     z_unique.sort()
#     return z_unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True, help="CFL3D plot3dg.bin")
    parser.add_argument("--q", required=True, help="CFL3D plot3dq.bin")
    #parser.add_argument("--target-hdf", required=True, help="SOD2D target .hdf mesh")
    parser.add_argument("--z-planes", type=int, required=True)
    parser.add_argument("--z-span", type=float, default=0.2)
    parser.add_argument("--porder", type=int, default=2)
    parser.add_argument("--out-src-mesh", default="cfl3d_source_12planes.h5")
    parser.add_argument("--out-src-restart", default="restart_cfl3d_source_23planes_1.h5")
    parser.add_argument("--mach", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--real", choices=["float32", "float64"], default="float64")
    parser.add_argument("--rho-constant-one", action="store_true")
    args = parser.parse_args()

    real_dtype = np.float32 if args.real == "float32" else np.float64

    grid_blocks = read_plot3d_grid_bin(args.grid, real_dtype)
    q_blocks = read_plot3d_q_bin(args.q, real_dtype)

    if len(grid_blocks) != len(q_blocks):
        raise RuntimeError("Grid and Q files have different number of blocks.")

    #z_target = get_target_z_planes(args.target_hdf)
    z_target = generate_z_planes(args.z_planes, args.z_span, args.porder)
    
    print("Target z planes:", z_target)
    print("Number of target z planes:", len(z_target))

    all_x = []
    all_y = []
    all_z = []

    all_rho = []
    all_ux = []
    all_uy = []
    all_uz = []
    all_p = []

    M = args.mach
    gamma = args.gamma

    for b, ((x, y, z), (rho, rhou, rhov, rhow, rhoE)) in enumerate(zip(grid_blocks, q_blocks)):
        ni, nj, nk = x.shape

        if nk != 2:
            print(f"Warning: block {b} has nk={nk}, not 2.")

        # Average the two CFL3D spanwise planes.
        # For 2.5D CFL3D this is usually identical, but averaging is safer.
        x2 = np.mean(x, axis=2)
        y2 = np.mean(y, axis=2)

        rho2 = np.mean(rho, axis=2)
        rhou2 = np.mean(rhou, axis=2)
        rhov2 = np.mean(rhov, axis=2)
        rhow2 = np.mean(rhow, axis=2)
        rhoE2 = np.mean(rhoE, axis=2)

        u_cfl = rhou2 / rho2
        v_cfl = rhov2 / rho2
        w_cfl = rhow2 / rho2

        ##Compute pressure from total energy
        kinetic = 0.5 * rho2 * (u_cfl**2 + v_cfl**2 + w_cfl**2)
        p_abs = (gamma - 1.0) * (rhoE2 - kinetic)

        # CFL3D compressible nondim has U_inf = Mach if a_inf = 1.
        # SOD2D incompressible setup  uses v0 = U_inf = 1.
        ## 7. Velocity scaling from CFL3D to SOD2D
        ux_sod = u_cfl / M
        uy_sod = v_cfl / M
        uz_sod = w_cfl / M

        p_inf = 1.0 / (gamma * M**2)

        # SOD2D incompressible pressure scale ~ rho_inf * U_inf^2.
        #p_sod = (p_abs - p_inf) / (M**2)
        #p_sod = np.zeros_like(rho)         ## Option A (simple, but less physical)
        p_sod = p_abs - np.mean(p_abs)      ## Option B (better physics)

        if args.rho_constant_one:
            rho_sod = np.ones_like(rho2)
        else:
            rho_sod = rho2.copy()

        for zz in z_target:
            all_x.append(x2.ravel(order="F"))
            all_y.append(y2.ravel(order="F"))
            all_z.append(np.full(x2.size, zz))

            all_rho.append(rho_sod.ravel(order="F"))
            all_ux.append(ux_sod.ravel(order="F"))
            all_uy.append(uy_sod.ravel(order="F"))
            all_uz.append(uz_sod.ravel(order="F"))
            all_p.append(p_sod.ravel(order="F"))

    X = np.concatenate(all_x)
    Y = np.concatenate(all_y)
    Z = np.concatenate(all_z)

    rho_out = np.concatenate(all_rho)
    ux_out = np.concatenate(all_ux)
    uy_out = np.concatenate(all_uy)
    uz_out = np.concatenate(all_uz)
    p_out = np.concatenate(all_p)

    with h5py.File(args.out_src_mesh, "w") as h5:
        grp = h5.create_group("Coords")
        grp.create_dataset("X", data=X)
        grp.create_dataset("Y", data=Y)
        grp.create_dataset("Z", data=Z)

    with h5py.File(args.out_src_restart, "w") as h5:
        h5.create_dataset("rho", data=rho_out)
        h5.create_dataset("u_x", data=ux_out)
        h5.create_dataset("u_y", data=uy_out)
        h5.create_dataset("u_z", data=uz_out)
        h5.create_dataset("p", data=p_out)
        h5.create_dataset("mue", data=np.zeros_like(rho_out))
        h5.create_dataset("time", data=np.array([0.0]))
        h5.create_dataset("istep", data=np.array([0], dtype=np.int32))

    print("Created source mesh   :", args.out_src_mesh)
    print("Created source restart:", args.out_src_restart)
    print("Total source points   :", X.size)
    print("u_x range:", ux_out.min(), ux_out.max())
    print("u_y range:", uy_out.min(), uy_out.max())
    print("u_z range:", uz_out.min(), uz_out.max())
    print("p range  :", p_out.min(), p_out.max())


if __name__ == "__main__":
    main()