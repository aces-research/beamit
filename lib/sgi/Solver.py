
import numpy as np

class NewtonSolver:

    def __init__(self, assembler):
        self.assembler = assembler
        self.Nmax = 20
        # the matrix
        self.A = np.empty([assembler.nequations, assembler.nequations])
        # the right hand side 
        self.f = np.empty([assembler.nequations])

    def solve(self):

        # nonlinear iterations
        for i in range(0, self.Nmax):
            # assemble the linear system
            self.assembler.assemble(self.A, self.f)
            # solve the linear system

            # assess convergence