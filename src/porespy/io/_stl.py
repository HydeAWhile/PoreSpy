import numpy as np
import pyvista as pv
import scipy.ndimage as spim
import skimage.measure as ms
from porespy.tools import sanitize_filename


__all__ = [
    "from_stl",
    "to_stl",
    "voxels_to_stl",
]


def _shell(stl_path, density, fill_holes=False):
    r"""
    Converts an STL file to a Numpy boolean array using Open3D shell voxelization

    Parameters
    ----------
    stl_path : str or path object
        The location and name of the STL file
    density : int 
        Controls the resolution of the final image. This is the number of
        voxels along the longest axis of the bounding box, with higher
        values leading to more voxels and hence better resolution. 
    fill_holes : bool (default=False)
        If `True` then steps are taken to ensure the mesh has no holes.
        This process can be slow so is not performed unless requested.

    Returns
    -------
    im : ndarray
        A boolean array with `True` values indicating the solid phase.

    Notes
    -----
    This method uses Open3D's ``VoxelGrid.create_from_triangle_mesh`` to
    voxelize the surface shell, then fills the interior with
    ``scipy.ndimage.binary_fill_holes``.  It is well suited for converting
    packings of particles (e.g. from DEM) into voxel images but might not
    work well on more nebulous shapes.
    """
    try:
        import open3d as o3d
    except ImportError:
            raise ImportError('open3d must be installed to use this function, which requires Python<=3.12')

    # Read the mesh
    mesh = o3d.io.read_triangle_mesh(stl_path)

    # Ensure the mesh is watertight
    mesh.compute_vertex_normals()
    mesh.orient_triangles()

    # Fill holes if any
    if fill_holes:
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        mesh = mesh.fill_holes()
        mesh = mesh.to_legacy()

    # Compute voxel size based on resolution
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_max_bound() - bbox.get_min_bound()
    voxel_size = np.max(extent) / density

    # Create voxel grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
        mesh,
        voxel_size=voxel_size
    )

    # Get voxels and convert to numpy array
    voxels = voxel_grid.get_voxels()

    if not voxels:  # Check if voxels list is empty
        return np.array([], dtype=bool)

    # Get the dimensions of the voxel grid
    voxel_indices = np.array([voxel.grid_index for voxel in voxels])
    grid_dims = voxel_indices.max(axis=0) + 1

    # Create boolean array and fill it
    voxel_array = np.zeros(grid_dims, dtype=bool)
    voxel_array[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = True

    # Fill inside of objects
    voxel_array = spim.binary_fill_holes(voxel_array)

    return voxel_array


def _enclosed(stl_path, density):
    r"""
    Converts an STL file to a boolean array using PyVista's enclosed-points test

    Parameters
    ----------
    stl_path : str or path object
        The location and name of the STL file
    density : int
        The number of voxels along the longest axis of the bounding box.
        Higher values give higher resolution but use more memory.

    Returns
    -------
    im : ndarray
        A boolean array with `True` values indicating the solid phase.

    Notes
    -----
    This method creates a regular grid over the mesh bounding box, then uses
    PyVista's ``select_enclosed_points`` to identify which grid points lie
    inside the surface.  It can be slow and memory-intensive for large meshes.
    """
    mesh = pv.read(stl_path)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    extent = max(x_max - x_min, y_max - y_min, z_max - z_min)
    voxel_size = extent / density
    x = np.arange(x_min, x_max, voxel_size)
    y = np.arange(y_min, y_max, voxel_size)
    z = np.arange(z_min, z_max, voxel_size)
    x, y, z = np.meshgrid(x, y, z)

    # Create unstructured grid from the structured grid
    grid = pv.StructuredGrid(x, y, z)
    ugrid = pv.UnstructuredGrid(grid)

    # Get part of the mesh within the mesh's bounding surface
    selection = ugrid.select_enclosed_points(
        mesh.extract_surface(),
        tolerance=0.0,
        check_surface=False,
    )
    mask = selection['SelectedPoints'].view(bool)
    mask = mask.reshape(x.shape, order='F')
    mask = np.array(mask)
    return mask


def _voxelize(stl_path, density):
    r"""
    Converts an STL file to a boolean array using PyVista's voxelize_volume

    Parameters
    ----------
    stl_path : str or path object
        The location and name of the STL file
    density : int
        The number of voxels along the longest axis of the bounding box.
        Higher values give higher resolution but use more memory.

    Returns
    -------
    im : ndarray
        A boolean array with `True` values indicating the solid phase.

    Notes
    -----
    This method uses PyVista's ``voxelize_volume`` which internally creates a
    volumetric representation of the mesh at the given density.
    """
    mesh = pv.read(stl_path)
    bounds = mesh.bounds
    extent = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
    voxel_size = extent / density
    vox = pv.voxelize_volume(mesh, density=voxel_size, check_surface=False)
    im = np.reshape(
        vox['InsideMesh'],
        np.array(vox.meshgrid[0].shape)-1,
        order='F',
    )
    return im.astype(bool)


def from_stl(stl_path, method='shell', density=100, fill_holes=False):
    r"""
    Converts an STL file to a boolean array

    Parameters
    ----------
    stl_path : str or path object
        The location and name of the STL file
    method : str
        The voxelization method to use.  Options are:

        ============ ===========================================================
        method       Description
        ============ ===========================================================
        ``'shell'``  Uses Open3D to voxelize the surface shell, then fills the
                     interior. This is the default and generally the fastest.
        ``'voxelize'`` Uses PyVista's ``voxelize_volume`` to create a volumetric
                     representation of the mesh.
        ``'enclosed'`` Uses PyVista's ``select_enclosed_points`` to test every
                     grid point against the surface. Can be slow and
                     memory-intensive for large meshes.
        ============ ===========================================================

    density : int
        The number of voxels along the longest axis of the bounding box.
        Higher values give higher resolution but use more memory. The default
        is 100 for all methods.
    fill_holes : bool
        If `True` then steps are taken to ensure the mesh has no holes before
        voxelization. Only used when ``method='shell'``. The default is
        `False`.

    Returns
    -------
    im : ndarray
        A boolean array with `True` values indicating the solid phase.

    """
    
    if method == 'shell':
        im = _shell(stl_path=stl_path, density=density, fill_holes=fill_holes)
    elif method == 'voxelize':
        im = _voxelize(stl_path=stl_path, density=density)
    elif method == 'enclosed':
        im = _enclosed(stl_path=stl_path, density=density)
    else:
        raise ValueError(
            f"Unsupported method '{method}'. "
            "Options are 'shell', 'voxelize', or 'enclosed'."
        )
    return im


def to_stl(im, filename, divide=False, downsample=False, voxel_size=1, vox=False):
    r"""
    Converts an array to an STL file.

    Parameters
    ----------
    im : 3D image
        The image of the porous material
    path : string
        Path to output file
    divide : bool
        vtk files can get very large, this option allows you for two output
        files, divided at z = half. This allows for large data sets to be
        imaged without loss of information
    downsample : bool
        very large images can be downsampled to half the size in each
        dimension, this doubles the effective voxel size
    voxel_size : int
        The side length of the voxels (voxels  are cubic)
    vox : bool
        For an image that is binary (1's and 0's) this reduces the file size by
        using int8 format (can also be used to reduce file size when accuracy
        is not necessary ie: just visulization)

    Notes
    -----
    Outputs an STL file that can opened in Paraview

    Examples
    --------
    `Click here
    <https://porespy.org/examples/io/reference/to_stl.html>`_
    to view online example.

    """
    filename = sanitize_filename(filename, ext="stl", exclude_ext=True)
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
    if im.dtype == bool:
        vox = True
    if vox:
        im = im.astype(np.int8)
    vs = voxel_size
    if divide:
        split = np.round(im.shape[2] / 2).astype(np.int)
        im1 = im[:, :, 0:split]
        im2 = im[:, :, split:]
        _save_stl(im1, vs, f"{filename}_1")
        _save_stl(im2, vs, f"{filename}_2")
    elif downsample:
        im = spim.interpolation.zoom(im, zoom=0.5, order=0, mode="reflect")
        _save_stl(im, vs * 2, filename)
    else:
        _save_stl(im, vs, filename)


def _save_stl(im, vs, filename):
    r"""
    Helper method to convert an array to an STL file.

    Parameters
    ----------
    im : 3D image
        The image of the porous material
    voxel_size : int
        The side length of the voxels (voxels are cubic)
    filename : string
        Path to output file

    """
    try:
        from stl import mesh
    except ModuleNotFoundError:
        msg = 'numpy-stl can be installed with pip install numpy-stl'
        raise ModuleNotFoundError(msg)
    im = np.pad(im, pad_width=10, mode="constant", constant_values=True)
    vertices, faces, norms, values = ms.marching_cubes(im)
    vertices *= vs
    # Export the STL file
    export = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            export.vectors[i][j] = vertices[f[j], :]
    export.save(f"{filename}.stl")


def voxels_to_stl(
    voxel_array,
    voxel_size=1.0,
    output_path=None,
    simplify=True,
    smooth=0,
):
    """
    Alternative version of to_stl that uses Open3d functions to simplify and
    smooth mesh.

    Parameters
    ----------
    voxel_array : ndarray
        3D boolean array of voxels
    voxel_size : float
        Size of each voxel
    output_path : str
        Path to save the STL file. If `None` (default) then no file is written, and
        only a the mesh object is returned.
    simplify : bool
        Whether to simplify the mesh
    smooth : int
        The number of times to smooth the mesh. The default is 0.

    Returns
    -------
    mesh : open3d.geometry.TriangleMesh
        The resulting mesh.
    """
    try:
        import open3d as o3d
    except ImportError:
            raise ImportError('open3d must be installed to use this function, which requires Python<=3.12')

    # Get initial mesh
    verts, faces, normals, values = ms.marching_cubes(voxel_array)

    # Scale vertices by voxel size
    verts = verts * voxel_size

    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    if simplify:
        # Simplify mesh while preserving shape
        mesh = mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average
        )

    if smooth:
        # Smooth the mesh
        mesh = mesh.filter_smooth_simple(number_of_iterations=smooth)

    # Ensure normals are computed
    mesh.compute_vertex_normals()

    # Optional: Remove any duplicate vertices
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()

    # Save to STL
    if output_path:
        o3d.io.write_triangle_mesh(output_path, mesh)

    return mesh
