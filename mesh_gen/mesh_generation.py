import gmsh
import sys

# Initialize the Gmsh API
gmsh.initialize()
gmsh.model.add("dealII_domain")

# --- 1. Define Geometry (e.g., Flow past a cylinder) ---
# Box (Fluid Domain)
L, H = 2.0, 1.0
p1 = gmsh.model.geo.addPoint(0, 0, 0)
p2 = gmsh.model.geo.addPoint(L, 0, 0)
p3 = gmsh.model.geo.addPoint(L, H, 0)
p4 = gmsh.model.geo.addPoint(0, H, 0)

l1 = gmsh.model.geo.addLine(p1, p2) # Bottom
l2 = gmsh.model.geo.addLine(p2, p3) # Outlet
l3 = gmsh.model.geo.addLine(p3, p4) # Top
l4 = gmsh.model.geo.addLine(p4, p1) # Inlet

# following function just merges together the lines created in order to form a closed border
box_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])

# Cylinder Points (adding random numbers)
cx, cy, r = 0.5, 0.5, 0.1
c1 = gmsh.model.geo.addPoint(cx, cy, 0)
c2 = gmsh.model.geo.addPoint(cx + r, cy, 0)
c3 = gmsh.model.geo.addPoint(cx - r, cy, 0)
c4 = gmsh.model.geo.addPoint(cx, cy + r, 0)
c5 = gmsh.model.geo.addPoint(cx, cy - r, 0)

# First argument of addCircleArc is the starting point, last one is the ending point. Middle one is the center of rotation
circle1 = gmsh.model.geo.addCircleArc(c2, c1, c4)
circle2 = gmsh.model.geo.addCircleArc(c4, c1, c3)
circle3 = gmsh.model.geo.addCircleArc(c3, c1, c5)
circle4 = gmsh.model.geo.addCircleArc(c5, c1, c2)

circle_loop = gmsh.model.geo.addCurveLoop([circle1, circle2, circle3, circle4])

# Define the Surface: 
#   - First element defines the outer boundary
#   - From the second element can define holes in the surface that will be erased 
surface = gmsh.model.geo.addPlaneSurface([box_loop, circle_loop])

gmsh.model.geo.synchronize()

# "Fields" are used to define triangulations dimensions given the distance from the cilinder
# this is true because close to the cylinder, gradients of u and p will modify at high frequency
# so we need smaller triangulations close to it and bigger ones (where the flow is more steady).
dist_field = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(dist_field, "CurvesList", [circle1, circle2, circle3, circle4])
gmsh.model.mesh.field.setNumber(dist_field, "Sampling", 100)

# 2. Create a threshold field: 
#    - Inside DistMin: Mesh size = LcMin (fine)
#    - Outside DistMax: Mesh size = LcMax (coarse)
threshold_field = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold_field, "IField", dist_field)
gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", 0.02) # Fine mesh near cylinder
gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.1)  # Coarse mesh far away
gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.15)
gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 0.5)

"""
As of explanations I got from documentations + LLMs fields actually define how the mesh is created.
If a point as to be added it first checks Layer 1 which calculates distance from the curves of the cylinder.
The result returned by this check is than passed by Layer 2 that calculates how to create the triangulation
given the distance. (dont ask me why but the strings used in this case are actually usefull to tell gmsh how to operate)  
"""

gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

# Same thing here with these physical groups which are used to tell dealII "what is what".
# I dont really get precisely how it works because I dont trust LLMs so much and documentation is a mess.
gmsh.model.addPhysicalGroup(1, [l4], 0, name="Inlet")
gmsh.model.addPhysicalGroup(1, [l2], 1, name="Outlet")
gmsh.model.addPhysicalGroup(1, [l1, l3], 2, name="Walls")
gmsh.model.addPhysicalGroup(1, [circle1, circle2, circle3, circle4], 3, name="Cylinder")

# Material ID = 0 indicates the fluid domain
# First argument of addPhysicalGroup specifies the dimension of the group so generally:
#   - 0 = A Point
#   - 1 = A Boundary
#   - 2 = A Surface
#   - 3 = A Volume (for 3D)
gmsh.model.addPhysicalGroup(2, [surface], 0, name="Fluid")

# --- 4. Generate and Export ---
gmsh.model.mesh.generate(2) # Generate 2D mesh
gmsh.write("./mesh/dealII_mesh.msh") # Export to .msh format

# Launch GUI to see the result (optional)
gmsh.fltk.run()

gmsh.finalize()