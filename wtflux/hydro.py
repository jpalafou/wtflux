from typing import Optional, Tuple, cast

from . import ArrayLike, cp, fuse


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
    return cp.sqrt(cp.where(sq_sound_speed > min_c2, sq_sound_speed, min_c2))


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


def _upwind(yl: ArrayLike, yr: ArrayLike, v: ArrayLike) -> ArrayLike:
    """
    Upwinding operator for states yl and yr with velocity v.

    Args:
        yl (ArrayLike): Left state.
        yr (ArrayLike): Right state.
        v (ArrayLike): Velocity.

    Returns:
        ArrayLike: Flux.
    """
    return v * cp.where(v > 0, yl, cp.where(v < 0, yr, 0))


@fuse
def advection_upwinding(
    rho_L: ArrayLike,
    v1_L: ArrayLike,
    v2_L: ArrayLike,
    v3_L: ArrayLike,
    rho_R: ArrayLike,
    v1_R: ArrayLike,
    v2_R: ArrayLike,
    v3_R: ArrayLike,
    passives_L: Optional[ArrayLike] = None,
    passives_R: Optional[ArrayLike] = None,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, Optional[ArrayLike]]:
    """
    Compute the advection upwinding fluxes.

    Args:
        rho_L (ArrayLike): Density on the left side.
        v1_L (ArrayLike): Principle velocity component on the left side.
        v2_L (ArrayLike): First transverse velocity component on the left side.
        v3_L (ArrayLike): Second transverse velocity component on the left side.
        rho_R (ArrayLike): Density on the right side.
        v1_R (ArrayLike): Principle velocity component on the right side.
        v2_R (ArrayLike): First transverse velocity component on the right side.
        v3_R (ArrayLike): Second transverse velocity component on the right side.
        passives_L (Optional[ArrayLike]): Primitive passive scalars on the left side.
        passives_R (Optional[ArrayLike]): Primitive passive scalars on the right side.

    Returns:
        Tuple[ArrayLike, ...]: Advection upwinding fluxes.
        - F_rho (ArrayLike): Density flux.
        - F_v1 (ArrayLike): Principle velocity flux.
        - F_v2 (ArrayLike): First transverse velocity flux.
        - F_v3 (ArrayLike): Second transverse velocity flux.
        - F_passives (Optional[ArrayLike]): Passive scalars flux with shape
            (n_passives, ...). None if either `passives_L` or `passives_R` are None.
    """
    v = cp.where(cp.abs(v1_L) > cp.abs(v1_R), v1_L, v1_R)
    F_rho = _upwind(rho_L, rho_R, v)
    F_v1 = _upwind(v1_L, v1_R, v)
    F_v2 = _upwind(v2_L, v2_R, v)
    F_v3 = _upwind(v3_L, v3_R, v)
    F_passives = (
        _upwind(passives_L, passives_R, v)
        if (passives_L is not None and passives_R is not None)
        else None
    )
    return F_rho, F_v1, F_v2, F_v3, F_passives


def _conservative_array(
    rho: ArrayLike,
    v1: ArrayLike,
    v2: ArrayLike,
    v3: ArrayLike,
    P: ArrayLike,
    gamma: float,
    passives: Optional[ArrayLike] = None,
    m1: Optional[ArrayLike] = None,
    m2: Optional[ArrayLike] = None,
    m3: Optional[ArrayLike] = None,
    E: Optional[ArrayLike] = None,
    conserved_passives: Optional[ArrayLike] = None,
) -> ArrayLike:
    """
    Create an array of conservative variables.

    Args:
        rho (ArrayLike): Density.
        v1 (ArrayLike): Velocity in a direction.
        v2 (ArrayLike): Velocity in another direction.
        v3 (ArrayLike): Velocity in yet another direction.
        P (ArrayLike): Pressure.
        gamma (float): Adiabatic index.
        passives (Optional[ArrayLike]): Primitive passive scalars.
        m1 (Optional[ArrayLike]): Momentum in a direction. If None, momentum will be
            computed from the primitive variables. Otherwise, the primitive variables
            are ignored.
        m2 (Optional[ArrayLike]): Momentum in another direction.
        m3 (Optional[ArrayLike]): Momentum in yet another direction.
        E (Optional[ArrayLike]): Total energy.
        conserved_passives (Optional[ArrayLike]): Conservative passives.

    Returns:
        U (ArrayLike): Array of conservative variables with shape (n_vars, ...).
        - U[0]: Density.
        - U[1]: Momentum in a direction.
        - U[2]: Momentum in another direction.
        - U[3]: Momentum in yet another direction.
        - U[4]: Total energy.
        - U[5:]: Conservative passives if `passives` is not None.
    """
    HAS_PASSIVES = passives is not None
    U = cp.empty(
        (5 + (cast(ArrayLike, passives).shape[0] if HAS_PASSIVES else 0),) + rho.shape,
        dtype=rho.dtype,
    )
    if m1 is not None:
        U[0] = rho
        U[1] = m1
        U[2] = m2
        U[3] = m3
        U[4] = E
    else:
        U[0], U[1], U[2], U[3], U[4], conserved_passives = (
            conservatives_from_primitives(rho, v1, v2, v3, P, gamma, passives)
        )
    if HAS_PASSIVES:
        U[5:] = conserved_passives
    return U


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
    passives_L: Optional[ArrayLike] = None,
    passives_R: Optional[ArrayLike] = None,
    m1_L: Optional[ArrayLike] = None,
    m2_L: Optional[ArrayLike] = None,
    m3_L: Optional[ArrayLike] = None,
    E_L: Optional[ArrayLike] = None,
    m1_R: Optional[ArrayLike] = None,
    m2_R: Optional[ArrayLike] = None,
    m3_R: Optional[ArrayLike] = None,
    E_R: Optional[ArrayLike] = None,
    conserved_passives_L: Optional[ArrayLike] = None,
    conserved_passives_R: Optional[ArrayLike] = None,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, Optional[ArrayLike]]:
    """
    Compute the Lax-Friedrichs fluxes to solve the Riemann problem between the L and R
    states.

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
        m1_L (Optional[ArrayLike]): Momentum in the principle direction on the left
            side. If None, momentum will be computed from the primitive variables.
        m2_L (Optional[ArrayLike]): Momentum in the first transverse direction on the
            left side.
        m3_L (Optional[ArrayLike]): Momentum in the second transverse direction on the
            left side.
        E_L (Optional[ArrayLike]): Total energy on the left side.
        m1_R (Optional[ArrayLike]): Momentum in the principle direction on the right
            side.
        m2_R (Optional[ArrayLike]): Momentum in the first transverse direction on the
            right side.
        m3_R (Optional[ArrayLike]): Momentum in the second transverse direction on the
            right side.
        E_R (Optional[ArrayLike]): Total energy on the right side.
        conserved_passives_L (Optional[ArrayLike]): Passive scalar density
            (rho*passives) on the left side.
        conserved_passives_R (Optional[ArrayLike]): Passive scalar density
            (rho*passives) on the right side.

    Returns:
        Tuple[ArrayLike, ...]: Lax-Friedrichs fluxes.
        - F_LF_rho (ArrayLike): Density flux.
        - F_LF_m1 (ArrayLike): Momentum flux in the principle direction.
        - F_LF_m2 (ArrayLike): Momentum flux in the first transverse direction.
        - F_LF_m3 (ArrayLike): Momentum flux in the second transverse direction.
        - F_LF_E (ArrayLike): Total energy flux.
        - F_LF_passives (Optional[ArrayLike]): Passive scalars flux with shape
            (n_passives, ...). None if either `passives_L` or `passives_R` are None.
    """
    # initialize flux arrays
    HAS_PASSIVES = passives_L is not None and passives_R is not None
    out_shape = (
        5 + (cast(ArrayLike, passives_L).shape[0] if HAS_PASSIVES else 0),
    ) + rho_L.shape
    out_type = rho_L.dtype
    F_L = cp.empty(out_shape, dtype=out_type)
    F_R = cp.empty(out_shape, dtype=out_type)

    # assign flux arrays
    F_L[0], F_L[1], F_L[2], F_L[3], F_L[4], F_passives_L = fluxes(
        rho_L, v1_L, v2_L, v3_L, P_L, gamma, passives_L
    )
    F_R[0], F_R[1], F_R[2], F_R[3], F_R[4], F_passives_R = fluxes(
        rho_R, v1_R, v2_R, v3_R, P_R, gamma, passives_R
    )
    if passives_L is not None and passives_R is not None:
        F_L[5:] = F_passives_L
        F_R[5:] = F_passives_R

    # compute the max wave speeds
    c_L = sound_speed(rho_L, P_L, gamma) + cp.abs(v1_L)
    c_R = sound_speed(rho_R, P_R, gamma) + cp.abs(v1_R)
    c_max = cp.maximum(c_L, c_R)

    # compute conservative variables if not provided
    U_L = _conservative_array(
        rho_L,
        v1_L,
        v2_L,
        v3_L,
        P_L,
        gamma,
        passives_L,
        m1_L,
        m2_L,
        m3_L,
        E_L,
        conserved_passives_L,
    )
    U_R = _conservative_array(
        rho_R,
        v1_R,
        v2_R,
        v3_R,
        P_R,
        gamma,
        passives_R,
        m1_R,
        m2_R,
        m3_R,
        E_R,
        conserved_passives_R,
    )

    # compute the Lax-Friedrichs fluxes
    F_LF = 0.5 * (F_L + F_R - c_max * (U_R - U_L))

    # return the fluxes
    return (
        F_LF[0],
        F_LF[1],
        F_LF[2],
        F_LF[3],
        F_LF[4],
        F_LF[5:] if HAS_PASSIVES else None,
    )
