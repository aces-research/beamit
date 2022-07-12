from sgi import FunctionSpace
from sgi import Material
from sgi import System
from sgi import Assembler
from sgi import Solver


def main():

    # elastic modulus of beam
    E = 1
    # area of cross section
    A = 1
    # physical information (material parameters)
    material = Material.Material(E, A)
    print("Created material")

    # length of beam
    L = 1
    # number of elements
    Nel = 10
    # geometric information (domain, elements, quadrature rule, shape functions)
    function_space = FunctionSpace.FunctionSpace(0, L, Nel)
    function_space.discretize()
    print("Discretized function space")

    # system as a 
    system = System.System(function_space, material)

    # the assembler (in charge of the continuous Galerkin stiffness and residual)
    assembler = Assembler.AssemblerCG(system)

    # the solver
    solver = Solver.NewtonSolver(assembler)
    # solve the nonlinear static problem and update the nodal position in system 
    solver.solve()

# run main function
main()