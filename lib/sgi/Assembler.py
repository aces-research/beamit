
class AssemblerCG:
    
    def __init__(self, system):
        # the number of equations
        # (for CG, this is the number of nodes * number of degrees of freedom per node)
        self.nequations = system.function_space.N * system.function_space.dof

    def assemble_stiffness(self, A):
        pass

    def assemble_residual(self, f):
        pass

    def assemble(self, A, f):
        # assemble stiffness
        self.assemble_stiffness(A)

        # assemble residual
        self.assemble_residual(f)