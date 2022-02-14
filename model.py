import numpy as np

class NeoHookeanSolutionGenerator:
    def __init__(self, omega, L, zeta, nu_f):
        self.omega = omega
        self.L = L
        self.zeta = zeta
        self.nu_f = nu_f
        # self.V_wall = V_wall

        self.rho_f = 1.0
        self.rho = 1.0

        (
            self.full_vf,
            self.full_vs,
            self.sensitivity_vf,
            self.sensitivity_vs,
        ) = self.construct_model()

    def construct_model(self):
        from sympy import (
            exp,
            I,
            conjugate,
            diff,
            lambdify,
            symbols,
            re,
            im,
            sin,
            cos,
            simplify,
            ln,
            sqrt,
            Matrix,
        )
        from sympy.solvers.solveset import linsolve

        y, t = symbols("y, t", real=True)
        A, B, C = symbols("A, B, C")
        # Re, Er = symbols('Re, Er')
        # nu = symbols('nu')
        c1 = symbols("c1")
        V_wall = symbols("V_wall")
        nu_s = symbols("nu_s")

        omega = self.omega
        rho_f = self.rho_f
        rho = self.rho  # fixed to one in this study
        rho_s = rho * rho_f

        # geometry
        zeta = self.zeta
        L = self.L
        L_s = zeta * 0.5 * L
        L_f = (1 - zeta) * 0.5 * L

        # derived
        # V_wall = gamma * omega * L / 2
        shear_rate = 2 * V_wall / omega / L

        # nu_f = shear_rate * omega * L_f ** 2 / Re
        nu_f = self.nu_f
        mu_f = nu_f * rho_f
        # nu_s = nu * nu_f
        nu = nu_s / nu_f
        mu_s = nu_s * rho_s
        # c1 = mu_f * shear_rate * omega / 2 / Er
        Re = shear_rate * omega * (L_f ** 2) / nu_f
        Er = mu_f * shear_rate * omega / 2.0 / c1

        lam_f = sqrt(1j * omega / nu_f)
        lam_s = omega / sqrt(1j * omega * nu_s + 2 * c1 / rho_s)

        # lam_f, lam_s = symbols('lambda_f, lambda_s')

        vel_f = (A * exp(lam_f * y) + B * exp(-lam_f * y)) * exp(I * omega * t)
        u_s = (C * sin(lam_s * y)) * exp(I * omega * t)
        vel_s = diff(u_s, t)

        k1, k2, k3, k4, k5, k6 = symbols("k1, k2, k3, k4, k5, k6")
        eq1 = A * k1 + B * k2 - 0.5
        eq2 = A * k3 + B * k4 - C * k5
        eq3 = A * k3 - B * k4 - C * k6
        (ans,) = linsolve([eq1, eq2, eq3], (A, B, C))

        ans = ans.subs(k1, exp(lam_f * (L_s + L_f)))
        ans = ans.subs(k2, exp(-lam_f * (L_s + L_f)))
        ans = ans.subs(k3, exp(lam_f * L_s))
        ans = ans.subs(k4, exp(-lam_f * L_s))
        ans = ans.subs(k5, 1j * omega * sin(lam_s * L_s))
        ans = ans.subs(
            k6, lam_s * cos(lam_s * L_s) / (mu_f * lam_f) * (2 * c1 + mu_s * 1j * omega)
        )

        vel_f = vel_f.subs(A, ans[0])  # .simplify()
        vel_f = vel_f.subs(B, ans[1])  # .simplify()
        vel_s = vel_s.subs(C, ans[2])  # .simplify()

        vel_f *= V_wall
        vel_s *= V_wall

        # local sensitivity matrices
        sensitivity_matrix_f = Matrix(
            [diff(vel_f, c1), diff(vel_f, V_wall), diff(vel_f, nu_s)]
        )
        sensitivity_matrix_s = Matrix(
            [diff(vel_s, c1), diff(vel_s, V_wall), diff(vel_s, nu_s)]
        )

        vel_f += conjugate(vel_f)
        vel_s += conjugate(vel_s)

        vel_fl = lambdify([c1, V_wall, nu_s, y, t], vel_f)
        vel_sl = lambdify([c1, V_wall, nu_s, y, t], vel_s)
        chi_fl = lambdify([c1, V_wall, nu_s, y, t], sensitivity_matrix_f)
        chi_sl = lambdify([c1, V_wall, nu_s, y, t], sensitivity_matrix_s)

        return vel_fl, vel_sl, chi_fl, chi_sl

    def generate_velocities_for(self, c1_val, V_wall_val, nu_s_val):
        def v_f(y, t):
            return np.real(self.full_vf(c1_val, V_wall_val, nu_s_val, y, t))

        def v_s(y, t):
            return np.real(self.full_vs(c1_val, V_wall_val, nu_s_val, y, t))

        return v_f, v_s

    def generate_sensitivities_for(self, c1_val, V_wall_val, nu_s_val):
        def chi_f(y, t):
            val = self.sensitivity_vf(c1_val, V_wall_val, nu_s_val, y, t)
            return np.real(val + np.conjugate(val))

        def chi_s(y, t):
            val = self.sensitivity_vs(c1_val, V_wall_val, nu_s_val, y, t)
            return np.real(val + np.conjugate(val))

        return chi_f, chi_s