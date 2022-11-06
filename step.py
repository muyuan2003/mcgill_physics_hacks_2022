import scipy.sparse.linalg as splinalg
import set_up as s
import computing as comp

def multiply_dimensions(tuple):
    result = 1

    length = len(tuple)
    for i in range(length):
        result *= tuple[i]

    return result


def step(velocities_prev, time_current):

        # Create Forces
        forces = s.create_forces_on_grid(
            time_current,
            s.coordinates,
        )

        # Step 1, add forces
        #               w1(x) = w0(x) + delta(t) f(x, t)
        velocities_forces_applied = (
                velocities_prev
                +
                s.dt
                *
                forces
        )

        # (2) Nonlinear convection (=self-advection)
        velocities_advected = comp.advect(
            field=velocities_forces_applied,
            vector_field=velocities_forces_applied,
        )

        # (3) Diffuse
        velocities_diffused = splinalg.cg(
            A=splinalg.LinearOperator(
                shape=( multiply_dimensions(s.vector_shape), multiply_dimensions(s.vector_shape) ),
                matvec=comp.diffusion_operator,
            ),
            b=velocities_advected.flatten(),
            maxiter=comp.max_iter,
        )[0].reshape(s.vector_shape)

        # (4.1) Compute a pressure correction
        pressure = splinalg.cg(
            A=splinalg.LinearOperator(
                shape=( multiply_dimensions(s.scalar_shape), multiply_dimensions(s.scalar_shape) ),
                matvec=comp.poisson_operator,
            ),
            b=comp.divergence(velocities_diffused).flatten(),
            maxiter=comp.max_iter,
        )[0].reshape(s.scalar_shape)

        # (4.2) Correct the velocities to be incompressible
        velocities_projected = (
                velocities_diffused
                -
                comp.gradient(pressure)
        )
        return velocities_projected