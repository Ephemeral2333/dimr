import os
import subprocess
import tempfile
import numpy as np
import trimesh
from trimesh.exchange.binvox import Binvox

def parse_binvox_header_fixed(fp):
    """
    Fixed version of parse_binvox_header that handles comment lines
    """
    line = fp.readline().strip()
    if hasattr(line, "decode"):
        binvox = b"#binvox"
        space = b" "
    else:
        binvox = "#binvox"
        space = " "
    if not line.startswith(binvox):
        raise OSError("Not a binvox file")
    
    # Skip comment lines until we find the dim line
    while True:
        line = fp.readline().strip()
        if line.startswith(b"dim" if hasattr(line, "decode") else "dim"):
            break
    
    shape = tuple(int(s) for s in line.split(space)[1:])
    translate = tuple(float(s) for s in fp.readline().strip().split(space)[1:])
    scale = float(fp.readline().strip().split(space)[1])
    fp.readline()  # Skip 'data' line
    return shape, translate, scale

def parse_binvox_fixed(fp, writeable=False):
    """
    Fixed version of parse_binvox that handles comment lines
    """
    shape, translate, scale = parse_binvox_header_fixed(fp)
    
    # Read the binary RLE data
    rle_data = fp.read()
    
    return Binvox(rle_data=rle_data, shape=shape, translate=translate, scale=scale)

def load_binvox_fixed(file_obj):
    """
    Fixed version of load_binvox
    """
    if hasattr(file_obj, "read"):
        data = parse_binvox_fixed(file_obj, writeable=True)
    else:
        with open(file_obj, "rb") as f:
            data = parse_binvox_fixed(f, writeable=True)
    
    # Convert RLE data to voxel grid
    voxels = np.zeros(data.shape, dtype=bool)
    
    # Parse binary RLE data
    rle_data = data.rle_data
    
    # Binary RLE decoding
    i = 0
    j = 0
    while j < len(rle_data) and i < voxels.size:
        count = rle_data[j]
        j += 1
        if j < len(rle_data):
            value = rle_data[j]
            j += 1
            for k in range(count):
                if i < voxels.size:
                    voxels.flat[i] = (value != 0)
                    i += 1
    
    # Create VoxelGrid object
    from trimesh.voxel import VoxelGrid
    voxel_grid = VoxelGrid(voxels)
    
    return voxel_grid

def voxelize_mesh_fixed(mesh, dimension=256, wireframe=False, dilated_carving=False, exact=False, verbose=False):
    """
    Fixed version of voxelize_mesh that uses xvfb-run for headless operation
    """
    if binvox_encoder is None:
        raise ValueError("binvox executable not found in PATH")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Export mesh to temporary file
        mesh_file = os.path.join(temp_dir, "mesh.ply")
        mesh.export(mesh_file)
        
        # Build binvox command
        cmd = ["xvfb-run", "-a", binvox_encoder]
        
        if wireframe:
            cmd.append("-aw")
        if dilated_carving:
            cmd.append("-dc")
        if exact:
            cmd.append("-e")
        else:
            cmd.append("-c")
            cmd.append("-v")
        
        cmd.extend(["-d", str(dimension), "-t", "binvox", mesh_file])
        
        if verbose:
            print(f"Running command: {' '.join(cmd)}")
        
        # Run binvox
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_dir)
        
        if result.returncode != 0:
            raise RuntimeError(f"binvox failed: {result.stderr}")
        
        # Read the generated binvox file
        binvox_file = os.path.join(temp_dir, "mesh.binvox")
        if not os.path.exists(binvox_file):
            raise RuntimeError("binvox did not generate output file")
        
        # Load the binvox file using our fixed parser
        voxels = load_binvox_fixed(binvox_file)
        
        return voxels

# Get the binvox executable path
binvox_encoder = None
for path in os.environ.get("PATH", "").split(os.pathsep):
    binvox_path = os.path.join(path, "binvox")
    if os.path.isfile(binvox_path) and os.access(binvox_path, os.X_OK):
        binvox_encoder = binvox_path
        break 