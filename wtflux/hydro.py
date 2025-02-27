from typing import Callable, Optional, Tuple, cast

from . import ArrayLike, fuse, xp


@fuse
def sound_speed(
    rho: ArrayLike,
    P: ArrayLike,
    gamma: float,
    min_c2: float = 1e-16,
) -> ArrayLike:
    """
    Compute the sound speed.

    Args:
        rho (ArrayLike): Density.
        P (ArrayLike): Pressure.
        gamma (float): Adiabatic index.
        min_c2 (float): Minimum allowed square of the sound speed.

    Returns:
        ArrayLike: Sound speed.
    """
    sq_sound_speed = gamma * P / rho
    return xp.sqrt(xp.where(sq_sound_speed > min_c2, sq_sound_speed, min_c2))


@fuse
def primitives_from_conservatives(
    rho: ArrayLike,
    m1: ArrayLike,
    m2: ArrayLike,
    m3: ArrayLike,
    E: ArrayLike,
    gamma: float,
    conserved_passives: Optional[ArrayLike] = None,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, Optional[ArrayLike]]:
    """
    Compute the primitive variables from the conservative variables.

    Args:
        rho (ArrayLike): Density.
        m1 (ArrayLike): Momentum in a direction.
        m2 (ArrayLike): Momentum in another direction.
        m3 (ArrayLike): Momentum in yet another direction.
        E (ArrayLike): Total energy.
        gamma (float): Adiabatic index.
        conserved_passives (Optional[ArrayLike]): Passive scalars density
            (rho*passives).

    Returns:
        Tuple[ArrayLike, ...]: Primitive variables.
        - rho (ArrayLike): Density.
        - v1 (ArrayLike): Velocity in a direction.
        - v2 (ArrayLike): Velocity in another direction.
        - v3 (ArrayLike): Velocity in yet another direction.
        - P (ArrayLike): Pressure.
        - passives (Optional[ArrayLike]): Primitive passives. None if `passives` is
            None.
    """
    v1 = m1 / rho
    v2 = m2 / rho
    v3 = m3 / rho
    K = 0.5 * rho * (v1**2 + v2**2 + v3**2)
    P = (gamma - 1) * (E - K)
    passives = conserved_passives / rho if conserved_passives is not None else None
    return rho, v1, v2, v3, P, passives


@fuse
def conservatives_from_primitives(
    rho: ArrayLike,
    v1: ArrayLike,
    v2: ArrayLike,
    v3: ArrayLike,
    P: ArrayLike,
    gamma: float,
    passives: Optional[ArrayLike] = None,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, Optional[ArrayLike]]:
    """
    Compute the conservative variables from the primitive variables.

    Args:
        rho (ArrayLike): Density.
        v1 (ArrayLike): Velocity in a direction.
        v2 (ArrayLike): Velocity in another direction.
        v3 (ArrayLike): Velocity in yet another direction.
        P (ArrayLike): Pressure.
        gamma (float): Adiabatic index.
        passives (Optional[ArrayLike]): Primitive passive scalars.

    Returns:
        Tuple[ArrayLike, ...]: Conservative variables.
        - rho (ArrayLike): Density.
        - m1 (ArrayLike): Momentum in a direction.
        - m2 (ArrayLike): Momentum in another direction.
        - m3 (ArrayLike): Momentum in yet another direction.
        - E (ArrayLike): Total energy.
        - consserved_passives (Optional[ArrayLike]): Conservative passives
            (rho*passives). None if `passives` is None.
    """
    m1 = rho * v1
    m2 = rho * v2
    m3 = rho * v3
    K = 0.5 * rho * (v1**2 + v2**2 + v3**2)
    E = K + P / (gamma - 1)
    consserved_passives = passives * rho if passives is not None else None
    return rho, m1, m2, m3, E, consserved_passives


@fuse
def fluxes(
    rho: ArrayLike,
    v1: ArrayLike,
    v2: ArrayLike,
    v3: ArrayLike,
    P: ArrayLike,
    gamma: float,
    passives: Optional[ArrayLike] = None,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, Optional[ArrayLike]]:
    """
    Compute the hydro fluxes.

    Args:
        rho (ArrayLike): Density.
        v1 (ArrayLike): Principle velocity component.
        v2 (ArrayLike): First transverse velocity component.
        v3 (ArrayLike): Second transverse velocity component.
        P (ArrayLike): Pressure.
        gamma (float): Adiabatic index.
        passives (Optional[ArrayLike]): Primitive passive scalars.

    Returns:
        Tuple[ArrayLike, ...]: Fluxes.
        - F_rho (ArrayLike): Density flux.
        - F_m1 (ArrayLike): Momentum flux in the principle direction.
        - F_m2 (ArrayLike): Momentum flux in the first transverse direction.
        - F_m3 (ArrayLike): Momentum flux in the second transverse direction.
        - F_E (ArrayLike): Total energy flux.
        - F_passives (Optional[ArrayLike]): Passive scalars flux. None if `passives` is
            None.
    """
    F_rho = rho * v1
    F_m1 = rho * v1**2 + P
    F_m2 = rho * v1 * v2
    F_m3 = rho * v1 * v3
    K = 0.5 * rho * (v1**2 + v2**2 + v3**2)
    F_E = (K + P / (gamma - 1) + P) * v1
    F_passives = passives * v1 if passives is not None else None
    return F_rho, F_m1, F_m2, F_m3, F_E, F_passives


@fuse
def call_riemann_solver(
    rs: Callable[..., Tuple[ArrayLike, ...]],
    rho_L: ArrayLike,
    v1_L: ArrayLike,
    v2_L: ArrayLike,
    v3_L: ArrayLike,
    P_L: ArrayLike,
    rho_R: ArrayLike,
    v1_R: ArrayLike,
    v2_R: ArrayLike,
    v3_R: ArrayLike,
    P_R: ArrayLike,
    gamma: float,
    m1_L: Optional[ArrayLike] = None,
    m2_L: Optional[ArrayLike] = None,
    m3_L: Optional[ArrayLike] = None,
    E_L: Optional[ArrayLike] = None,
    m1_R: Optional[ArrayLike] = None,
    m2_R: Optional[ArrayLike] = None,
    m3_R: Optional[ArrayLike] = None,
    E_R: Optional[ArrayLike] = None,
    passives_L: Optional[ArrayLike] = None,
    passives_R: Optional[ArrayLike] = None,
    conserved_passives_L: Optional[ArrayLike] = None,
    conserved_passives_R: Optional[ArrayLike] = None,
) -> Tuple[ArrayLike, ...]:
    COMPUTE_PRIMITIVES = m1_L is None
    if COMPUTE_PRIMITIVES:
        _, m1_L, m2_L, m3_L, E_L, conserved_passives_L = conservatives_from_primitives(
            rho_L, v1_L, v2_L, v3_L, P_L, gamma, passives_L
        )
        _, m1_R, m2_R, m3_R, E_R, conserved_passives_R = conservatives_from_primitives(
            rho_R, v1_R, v2_R, v3_R, P_R, gamma, passives_R
        )
    return rs(
        rho_L,
        v1_L,
        v2_L,
        v3_L,
        P_L,
        rho_R,
        v1_R,
        v2_R,
        v3_R,
        P_R,
        gamma,
        m1_L,
        m2_L,
        m3_L,
        E_L,
        m1_R,
        m2_R,
        m3_R,
        E_R,
        passives_L,
        passives_R,
        conserved_passives_L,
        conserved_passives_R,
    )


@fuse
def _llf_operator(
    F_L: Optional[ArrayLike],
    F_R: Optional[ArrayLike],
    U_L: Optional[ArrayLike],
    U_R: Optional[ArrayLike],
    c_max: Optional[ArrayLike],
) -> Optional[ArrayLike]:
    return (
        0.5
        * (
            cast(ArrayLike, F_L)
            + cast(ArrayLike, F_R)
            - cast(ArrayLike, c_max) * (cast(ArrayLike, U_R) - cast(ArrayLike, U_L))
        )
        if F_L is not None
        else None
    )


@fuse
def _llf(
    rho_L: ArrayLike,
    v1_L: ArrayLike,
    v2_L: ArrayLike,
    v3_L: ArrayLike,
    P_L: ArrayLike,
    rho_R: ArrayLike,
    v1_R: ArrayLike,
    v2_R: ArrayLike,
    v3_R: ArrayLike,
    P_R: ArrayLike,
    gamma: float,
    m1_L: ArrayLike,
    m2_L: ArrayLike,
    m3_L: ArrayLike,
    E_L: ArrayLike,
    m1_R: ArrayLike,
    m2_R: ArrayLike,
    m3_R: ArrayLike,
    E_R: ArrayLike,
    passives_L: Optional[ArrayLike] = None,
    passives_R: Optional[ArrayLike] = None,
    conserved_passives_L: Optional[ArrayLike] = None,
    conserved_passives_R: Optional[ArrayLike] = None,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, Optional[ArrayLike]]:
    # assign flux arrays
    F_rho_L, F_m1_L, F_m2_L, F_m3_L, F_E_L, F_passives_L = fluxes(
        rho_L, v1_L, v2_L, v3_L, P_L, gamma, passives_L
    )
    F_rho_R, F_m1_R, F_m2_R, F_m3_R, F_E_R, F_passives_R = fluxes(
        rho_R, v1_R, v2_R, v3_R, P_R, gamma, passives_R
    )

    # compute the max wave speeds
    c_L = sound_speed(rho_L, P_L, gamma) + xp.abs(v1_L)
    c_R = sound_speed(rho_R, P_R, gamma) + xp.abs(v1_R)
    c_max = xp.maximum(c_L, c_R)

    # compute the Lax-Friedrichs fluxes
    F_rho = _llf_operator(F_rho_L, F_rho_R, rho_L, rho_R, c_max)
    F_m1 = _llf_operator(F_m1_L, F_m1_R, m1_L, m1_R, c_max)
    F_m2 = _llf_operator(F_m2_L, F_m2_R, m2_L, m2_R, c_max)
    F_m3 = _llf_operator(F_m3_L, F_m3_R, m3_L, m3_R, c_max)
    F_E = _llf_operator(F_E_L, F_E_R, E_L, E_R, c_max)
    F_conserved_passives = _llf_operator(
        F_passives_L, F_passives_R, conserved_passives_L, conserved_passives_R, c_max
    )

    # return the fluxes
    return F_rho, F_m1, F_m2, F_m3, F_E, F_conserved_passives


@fuse
def llf(
    rho_L: ArrayLike,
    v1_L: ArrayLike,
    v2_L: ArrayLike,
    v3_L: ArrayLike,
    P_L: ArrayLike,
    rho_R: ArrayLike,
    v1_R: ArrayLike,
    v2_R: ArrayLike,
    v3_R: ArrayLike,
    P_R: ArrayLike,
    gamma: float,
    m1_L: Optional[ArrayLike] = None,
    m2_L: Optional[ArrayLike] = None,
    m3_L: Optional[ArrayLike] = None,
    E_L: Optional[ArrayLike] = None,
    m1_R: Optional[ArrayLike] = None,
    m2_R: Optional[ArrayLike] = None,
    m3_R: Optional[ArrayLike] = None,
    E_R: Optional[ArrayLike] = None,
    passives_L: Optional[ArrayLike] = None,
    passives_R: Optional[ArrayLike] = None,
    conserved_passives_L: Optional[ArrayLike] = None,
    conserved_passives_R: Optional[ArrayLike] = None,
) -> Tuple[ArrayLike, ...]:
    """
    Prepare conservative variables for the Riemann solver and call it.

    Args:
        rho_L (ArrayLike): Density on the left side.
        v1_L (ArrayLike): Principle velocity component on the left side.
        v2_L (ArrayLike): First transverse velocity component on the left side.
        v3_L (ArrayLike): Second transverse velocity component on the left side.
        P_L (ArrayLike): Pressure on the left side.
        rho_R (ArrayLike): Density on the right side.
        v1_R (ArrayLike): Principle velocity component on the right side.
        v2_R (ArrayLike): First transverse velocity component on the right side.
        v3_R (ArrayLike): Second transverse velocity component on the right side.
        P_R (ArrayLike): Pressure on the right side.
        gamma (float): Adiabatic index.
        passives_L (Optional[ArrayLike]): Primitive passive scalars on the left side.
        passives_R (Optional[ArrayLike]): Primitive passive scalars on the right side.
        m1_L (ArrayLike): Momentum in the principle direction on the left side.
        m2_L (ArrayLike): Momentum in the first transverse direction on the left side.
        m3_L (ArrayLike): Momentum in the second transverse direction on the left side.
        E_L (ArrayLike): Total energy on the left side.
        m1_R (ArrayLike): Momentum in the principle direction on the right side.
        m2_R (ArrayLike): Momentum in the first transverse direction on the right side.
        m3_R (ArrayLike): Momentum in the second transverse direction on the right
            side.
        E_R (ArrayLike): Total energy on the right side.
        conserved_passives_L (Optional[ArrayLike]): Conservative passive scalars on the
            left side.
        conserved_passives_R (Optional[ArrayLike]): Conservative passive scalars on the
            right side.

    Returns:
        Tuple[ArrayLike, ...]: Fluxes.
    """
    return call_riemann_solver(
        _llf,
        rho_L,
        v1_L,
        v2_L,
        v3_L,
        P_L,
        rho_R,
        v1_R,
        v2_R,
        v3_R,
        P_R,
        gamma,
        m1_L,
        m2_L,
        m3_L,
        E_L,
        m1_R,
        m2_R,
        m3_R,
        E_R,
        passives_L,
        passives_R,
        conserved_passives_L,
        conserved_passives_R,
    )
