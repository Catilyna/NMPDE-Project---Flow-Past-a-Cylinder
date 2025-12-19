import gmsh
import argparse
import os

class Mesh3D:
    """
    Class used to initialize a 3D mesh for a Navier-Stokes equation problem.
    Uses the OpenCASCADE (OCC) kernel for Constructive Solid Geometry (CSG).
    """
    def __init__(self, L, H, W):
        self.L = L
        self.H = H
        self.W = W # Width of the channel
        
        # Initialize gmsh
        gmsh.initialize()
        gmsh.model.add("dealII_domain_3D")
        
        # Storage for tags
        self.volume_tag = None
        self.obstacle_surface_tags = []
        self.outer_surface_tags = []

    def create_geometry(self, cx, cy, r):
        """
        Creates a Box and subtracts a Cylinder from it using Boolean operations.
        """
        # Create the main Channel
        # Arguments: x, y, z, dx, dy, dz
        channel_id = gmsh.model.occ.addBox(0, 0, 0, self.L, self.H, self.W)

        # Create the Cylinder
        # Arguments: x, y, z, dx, dy, dz, r
        # Cylinder runs along the Z-axis
        cylinder_id = gmsh.model.occ.addCylinder(cx, cy, 0, 0, 0, self.W, r)

        # We subtract the cylinder from the channel
        cut_result = gmsh.model.occ.cut([(3, channel_id)], [(3, cylinder_id)])
        
        # The result of the cut is our fluid domain volume
        self.volume_tag = cut_result[0][0][1] # cur_result are stored like tuples inside a list with ((outputDimTags), (outputDimTagsMap)) so it requires 3 indices to access elements

        # 4. Synchronize: Important when using OCC kernel to pass data to GMSH model
        gmsh.model.occ.synchronize()

    def set_physical_groups(self):
        """
        Automatically detects surfaces based on their geometric location 
        to assign Physical Groups (Inlet, Outlet, Walls, Obstacle).
        """
        surfaces = gmsh.model.getEntities(dim=2)
        
        inlet_tag = []
        outlet_tag = []
        walls_tags = []
        obstacle_tags = []
        
        eps = 1e-3

        for surface in surfaces:
            tag = surface[1]
            # Get the bounding box of the surface to determine its position
            bbox = gmsh.model.getBoundingBox(2, tag)
            # Logic to identify surfaces:            # bbox = [xmin, ymin, zmin, xmax, ymax, zmax]
            
            # tries to identify each surface given the values stored in com
            
            # Identify the Inlet
            if abs(bbox[0] - 0.0) < eps and abs(bbox[3] - 0.0) < eps:
                inlet_tag.append(tag)
                
            # Identify the Outlet
            elif abs(bbox[0] - self.L) < eps and abs(bbox[3] - self.L) < eps:
                outlet_tag.append(tag)
                
            # Checks wheter y=0 or y=H or z=0 or z=W
            elif ( (abs(bbox[1]) < eps and abs(bbox[4]) < eps) or       # Bottom (y=0)
                   (abs(bbox[1] - self.H) < eps and abs(bbox[4] - self.H) < eps) or   # Top (y=H)
                   (abs(bbox[2]) < eps and abs(bbox[5]) < eps) or       # Front (z=0)
                   (abs(bbox[2] - self.W) < eps and abs(bbox[5] - self.W) < eps) ):   # Back (z=W)
                walls_tags.append(tag)
            
            # If it's not the outer box, it must be the cylinder surface inside
            else:
                obstacle_tags.append(tag)

        # Create Physical Groups (Crucial for deal.II)
        # 1: Inlet, 2: Outlet, 3: Walls, 4: Obstacle
        if inlet_tag: gmsh.model.addPhysicalGroup(2, inlet_tag, 0, name="Inlet")
        if outlet_tag: gmsh.model.addPhysicalGroup(2, outlet_tag, 1, name="Outlet")
        if walls_tags: gmsh.model.addPhysicalGroup(2, walls_tags, 2, name="Walls")
        if obstacle_tags: gmsh.model.addPhysicalGroup(2, obstacle_tags, 3, name="Obstacle")
        
        # Physical Volume (0 usually used for material id)
        gmsh.model.addPhysicalGroup(3, [self.volume_tag], 0, name="Fluid")
        
        # Store obstacle tags for mesh refinement later
        self.obstacle_surface_tags = obstacle_tags

    def set_fields(self, distMin, distMax, lcMin, lcMax):
        """
        Sets mesh refinement fields based on distance from the obstacle.
        """
        # 1. Distance Field: Calculate distance from the obstacle surfaces
        dist_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_field, "FacesList", self.obstacle_surface_tags)
        gmsh.model.mesh.field.setNumber(dist_field, "Sampling", 100)

        # 2. Threshold Field: Refine based on the calculated distance
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "IField", dist_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", lcMin) # Fine mesh near cylinder
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", lcMax) # Coarse mesh far away
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", distMin)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", distMax)
        
        # Set as background mesh
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

    def build(self, cx, cy, r, output_filename, show_gui=False):
        self.create_geometry(cx, cy, r)
        
        self.set_physical_groups()
        
        # lcMin/Max might need to be larger for 3D to keep cell count reasonable and not stress RAM
        self.set_fields(distMin=0.15, distMax=0.5, lcMin=0.08, lcMax=0.3)
        
        gmsh.model.mesh.generate(3)
        
        output_dir = "../mesh"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        gmsh.write(f"{output_dir}/{output_filename}.msh")
        print(f"3D Mesh written to {output_dir}/{output_filename}.msh")

        gmsh.fltk.run()

        gmsh.finalize()

def main():
    parser = argparse.ArgumentParser(description="Generate a 3D mesh for deal.II")

    # Geometric Arguments
    parser.add_argument('-L', '--length', type=float, default=2.5, 
                        help='Length of the domain (x)')
    parser.add_argument('-H', '--height', type=float, default=0.41, 
                        help='Height of the domain (y)')
    parser.add_argument('-W', '--width', type=float, default=0.41, 
                        help='Width/Depth of the domain (z)')
    
    # Cylinder Position
    parser.add_argument('-cx', '--cx', type=float, default=0.5, 
                        help='Cylinder center X')
    parser.add_argument('-cy', '--cy', type=float, default=0.2, 
                        help='Cylinder center Y')
    parser.add_argument('-r', '--radius', type=float, default=0.05, 
                        help='Cylinder radius')
    
    args = parser.parse_args()
    
    mesh = Mesh3D(args.length, args.height, args.width)
    mesh.build(args.cx, args.cy, args.radius, "mesh3D_example")

if __name__ == "__main__":
    main()