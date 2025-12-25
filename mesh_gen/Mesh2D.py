import gmsh
import argparse

class Mesh2D:
    """
    Class used to inizialize mesh for a Navier Stokes equation problem
    """
    def __init__(self, L, H):
        self.l = L
        self.h = H
        
        self.lines = []
        self.circles = []
        
        # initialize gmsh
        gmsh.initialize()
        gmsh.model.add("dealII_domain")
        
        self.box_loop = None
        self.circle_loop = None
        
        self.surface = None
        
        
    def create_rect(self):
        p1 = gmsh.model.geo.addPoint(0, 0, 0)
        p2 = gmsh.model.geo.addPoint(self.l, 0, 0)
        p3 = gmsh.model.geo.addPoint(self.l, self.h, 0)
        p4 = gmsh.model.geo.addPoint(0, self.h, 0)
        
        self.lines.append(gmsh.model.geo.addLine(p1, p2))
        self.lines.append(gmsh.model.geo.addLine(p2, p3))
        self.lines.append(gmsh.model.geo.addLine(p3, p4))
        self.lines.append(gmsh.model.geo.addLine(p4, p1))
        
        # usefull to tell dealII what is what precisely
        gmsh.model.addPhysicalGroup(1, [self.lines[3]], 0, name="Inlet")
        gmsh.model.addPhysicalGroup(1, [self.lines[1]], 1, name="Outlet")
        gmsh.model.addPhysicalGroup(1, [self.lines[0], self.lines[2]], 2, name="Walls")
        
        self.box_loop = gmsh.model.geo.addCurveLoop(self.lines)
        
    def create_circle(self, cx, cy, r):
        """
        Function creates a circle given the (x, y) coordinates and the radius.
        """
        c1 = gmsh.model.geo.addPoint(cx, cy, 0)
        c2 = gmsh.model.geo.addPoint(cx + r, cy, 0)
        c3 = gmsh.model.geo.addPoint(cx - r, cy, 0)
        c4 = gmsh.model.geo.addPoint(cx, cy + r, 0)
        c5 = gmsh.model.geo.addPoint(cx, cy - r, 0)

        # First argument of addCircleArc is the starting point, last one is the ending point. Middle one is the center of rotation
        self.circles.append(gmsh.model.geo.addCircleArc(c2, c1, c4))
        self.circles.append(gmsh.model.geo.addCircleArc(c4, c1, c3))
        self.circles.append(gmsh.model.geo.addCircleArc(c3, c1, c5))
        self.circles.append(gmsh.model.geo.addCircleArc(c5, c1, c2))

        gmsh.model.addPhysicalGroup(1, self.circles, 3, name="Cylinder")

        self.circle_loop = gmsh.model.geo.addCurveLoop(self.circles)
        
    def setSurface(self):
        self.surface = gmsh.model.geo.addPlaneSurface([self.box_loop, self.circle_loop])
        gmsh.model.addPhysicalGroup(2, [self.surface], 0, name="Fluid")

    def setFields(self, distMin, distMax, lcMin, lcMax, sampling):   
        """ 'Fields' are used to define triangulations dimensions given the distance from the cilinder
            this is true because close to the cylinder, gradients of u and p will modify at high frequency
            so we need smaller triangulations close to it and bigger ones (where the flow is more steady). """
        dist_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_field, "CurvesList", self.circles)
        gmsh.model.mesh.field.setNumber(dist_field, "Sampling", sampling)

        # 2. Create a threshold field: 
        #    - Inside DistMin: Mesh size = LcMin (fine)
        #    - Outside DistMax: Mesh size = LcMax (coarse)
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "IField", dist_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", lcMin) # Fine mesh near cylinder
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", lcMax)  # Coarse mesh far away
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", distMin)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", distMax)
        
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
        
    def build(self, cx, cy, r, output_filename):
        self.create_rect()
        self.create_circle(cx, cy, r)
        
        self.setSurface()
        
        gmsh.model.geo.synchronize()
        
        self.setFields(0.0075, 0.25, 0.01, 0.05, 100)       
        
        gmsh.model.mesh.generate(2) # Generate 2D mesh
        gmsh.write(f"../mesh/2D/{output_filename}.msh")

        # Launch GUI to see the result (optional)
        gmsh.fltk.run()

        gmsh.finalize()
        
    
def main():
    parser = argparse.ArgumentParser(description="Generate a 2D rectangular mesh for deal.II")

    # Geometric Arguments
    parser.add_argument('-L', '--length', type=float, default=2.0, 
                        help='Length of the domain (default: 2.0)')
    parser.add_argument('-H', '--height', type=float, default=1.0, 
                        help='Height of the domain (default: 1.0)')
    
    parser.add_argument('-cx', '--cx', type=float, default=1.0, 
                        help='Set the cx of the center ot the cilinder')
    parser.add_argument('-cy', '--cy', type=float, default=0.5, 
                        help='Set the cy of the center ot the cilinder')
    parser.add_argument('-r', '--radius', type=float, default=0.1, 
                        help='Set the radius of the cilinder')
    parser.add_argument('-n', '--name', type=str, default="mesh2D", 
                        help='filename')

    # Parse the arguments
    args = parser.parse_args()
    
    mesh = Mesh2D(args.length, args.height)
    mesh.build(args.cx, args.cy, args.radius, args.name)


if __name__ == "__main__":
    main()

        
        
                