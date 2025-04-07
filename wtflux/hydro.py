from typing import Optional, Protocol, Tuple

from .backend import ArrayLike, fuse, xp


@fuse
def avoid_0(arr: ArrayLike, tol: float) -> ArrayLike:
    """
    Avoid division by zero.

    Args:
        arr (ArrayLike): Input array.
        tol (float): Tolerance.

    Returns:
        ArrayLike: Array with values below tol replaced by tol.
    """
    return xp.where(xp.abs(arr) < tol, xp.where(arr < tol, -tol, tol), arr)


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
    return xp.sqrt(xp.maximum(gamma * P / rho, min_c2))


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
        - conserved_passives (Optional[ArrayLike]): Conservative passives
            (rho*passives). None if `passives` is None.
    """
    m1 = rho * v1
    m2 = rho * v2
    m3 = rho * v3
    K = 0.5 * rho * (v1**2 + v2**2 + v3**2)
    E = K + P / (gamma - 1)
    conserved_passives = passives * rho if passives is not None else None
    return rho, m1, m2, m3, E, conserved_passives


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
        v1 (ArrayLike): Principal velocity component.
        v2 (ArrayLike): First transverse velocity component.
        v3 (ArrayLike): Second transverse velocity component.
        P (ArrayLike): Pressure.
        gamma (float): Adiabatic index.
        passives (Optional[ArrayLike]): Primitive passive scalars.

    Returns:
        Tuple[ArrayLike, ...]: Fluxes.
        - F_rho (ArrayLike): Density flux.
        - F_m1 (ArrayLike): Momentum flux in the principal direction.
        - F_m2 (ArrayLike): Momentum flux in the first transverse direction.
        - F_m3 (ArrayLike): Momentum flux in the second transverse direction.
        - F_E (ArrayLike): Total energy flux.
        - F_conserved_passives (Optional[ArrayLike]): Passive scalars flux. None if
            `passives` is None.
    """
    F_rho = rho * v1
    F_m1 = rho * v1**2 + P
    F_m2 = rho * v1 * v2
    F_m3 = rho * v1 * v3
    K = 0.5 * rho * (v1**2 + v2**2 + v3**2)
    F_E = (K + P / (gamma - 1) + P) * v1
    F_conserved_passives = rho * passives * v1 if passives is not None else None
    return F_rho, F_m1, F_m2, F_m3, F_E, F_conserved_passives


class RiemannSolverFn(Protocol):
    def __call__(
        self,
        rho_L: ArrayLike,
        v1_L: ArrayLike,
        v2_L: ArrayLike,
        v3_L: ArrayLike,
        P_L: ArrayLike,
        m1_L: ArrayLike,
        m2_L: ArrayLike,
        m3_L: ArrayLike,
        E_L: ArrayLike,
        rho_R: ArrayLike,
        v1_R: ArrayLike,
        v2_R: ArrayLike,
        v3_R: ArrayLike,
        P_R: ArrayLike,
        m1_R: ArrayLike,
        m2_R: ArrayLike,
        m3_R: ArrayLike,
        E_R: ArrayLike,
        gamma: float,
        passives_L: Optional[ArrayLike],
        conserved_passives_L: Optional[ArrayLike],
        passives_R: Optional[ArrayLike],
        conserved_passives_R: Optional[ArrayLike],
    ) -> Tuple[
        ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, Optional[ArrayLike]
    ]: ...


@fuse
def call_riemann_solver(
    rs: RiemannSolverFn,
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
    primitives: bool = True,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, Optional[ArrayLike]]:
    # prepare conservative variables
    if primitives:
        _, m1_L, m2_L, m3_L, E_L, conserved_passives_L = conservatives_from_primitives(
            rho_L, v1_L, v2_L, v3_L, P_L, gamma, passives_L
        )
        _, m1_R, m2_R, m3_R, E_R, conserved_passives_R = conservatives_from_primitives(
            rho_R, v1_R, v2_R, v3_R, P_R, gamma, passives_R
        )
    else:
        _, m1_L, m2_L, m3_L, E_L, conserved_passives_L = (
            rho_L,
            v1_L,
            v2_L,
            v3_L,
            P_L,
            passives_L,
        )
        _, m1_R, m2_R, m3_R, E_R, conserved_passives_R = (
            rho_R,
            v1_R,
            v2_R,
            v3_R,
            P_R,
            passives_R,
        )
        _, v1_L, v2_L, v3_L, P_L, passives_L = primitives_from_conservatives(
            rho_L, m1_L, m2_L, m3_L, E_L, gamma, conserved_passives_L
        )
        _, v1_R, v2_R, v3_R, P_R, passives_R = primitives_from_conservatives(
            rho_R, m1_R, m2_R, m3_R, E_R, gamma, conserved_passives_R
        )
    return rs(
        rho_L,
        v1_L,
        v2_L,
        v3_L,
        P_L,
        m1_L,
        m2_L,
        m3_L,
        E_L,
        rho_R,
        v1_R,
        v2_R,
        v3_R,
        P_R,
        m1_R,
        m2_R,
        m3_R,
        E_R,
        gamma,
        passives_L,
        conserved_passives_L,
        passives_R,
        conserved_passives_R,
    )


@fuse
def _llf_operator(
    F_L: ArrayLike,
    F_R: ArrayLike,
    U_L: ArrayLike,
    U_R: ArrayLike,
    c_max: ArrayLike,
) -> ArrayLike:
    """
    Lax-Friedrichs operator.
    """
    return 0.5 * (F_L + F_R - c_max * (U_R - U_L))


@fuse
def _llf(
    rho_L: ArrayLike,
    v1_L: ArrayLike,
    v2_L: ArrayLike,
    v3_L: ArrayLike,
    P_L: ArrayLike,
    m1_L: ArrayLike,
    m2_L: ArrayLike,
    m3_L: ArrayLike,
    E_L: ArrayLike,
    rho_R: ArrayLike,
    v1_R: ArrayLike,
    v2_R: ArrayLike,
    v3_R: ArrayLike,
    P_R: ArrayLike,
    m1_R: ArrayLike,
    m2_R: ArrayLike,
    m3_R: ArrayLike,
    E_R: ArrayLike,
    gamma: float,
    passives_L: Optional[ArrayLike],
    conserved_passives_L: Optional[ArrayLike],
    passives_R: Optional[ArrayLike],
    conserved_passives_R: Optional[ArrayLike],
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, Optional[ArrayLike]]:
    # assign flux arrays
    F_rho_L, F_m1_L, F_m2_L, F_m3_L, F_E_L, F_conserved_passives_L = fluxes(
        rho_L, v1_L, v2_L, v3_L, P_L, gamma, passives_L
    )
    F_rho_R, F_m1_R, F_m2_R, F_m3_R, F_E_R, F_conserved_passives_R = fluxes(
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
    if not any(x is None for x in [passives_L, passives_R]):
        F_conserved_passives = _llf_operator(
            F_conserved_passives_L,
            F_conserved_passives_R,
            conserved_passives_L,
            conserved_passives_R,
            c_max,
        )
    else:
        F_conserved_passives = None

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
    passives_L: Optional[ArrayLike] = None,
    passives_R: Optional[ArrayLike] = None,
    primitives: bool = True,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, Optional[ArrayLike]]:
    """
    Compute the Lax-Friedrichs fluxes.

    Args:
        rho_L (ArrayLike): Left density.
        v1_L (ArrayLike): Left principal velocity component.
        v2_L (ArrayLike): Left first transverse velocity component.
        v3_L (ArrayLike): Left second transverse velocity component.
        P_L (ArrayLike): Left pressure.
        rho_R (ArrayLike): Right density.
        v1_R (ArrayLike): Right principal velocity component.
        v2_R (ArrayLike): Right first transverse velocity component.
        v3_R (ArrayLike): Right second transverse velocity component.
        P_R (ArrayLike): Right pressure.
        gamma (float): Adiabatic index.
        passives_L (Optional[ArrayLike]): Left passive scalars density
            (rho*passives).
        passives_R (Optional[ArrayLike]): Right passive scalars density
            (rho*passives).
        primitives (bool): Whether the input variables are primitive variables. If
            False, they are considered conservative variables.
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
        passives_L,
        passives_R,
        primitives,
    )


@fuse
def _hllc(
    rho_L: ArrayLike,
    v1_L: ArrayLike,
    v2_L: ArrayLike,
    v3_L: ArrayLike,
    P_L: ArrayLike,
    m1_L: ArrayLike,
    m2_L: ArrayLike,
    m3_L: ArrayLike,
    E_L: ArrayLike,
    rho_R: ArrayLike,
    v1_R: ArrayLike,
    v2_R: ArrayLike,
    v3_R: ArrayLike,
    P_R: ArrayLike,
    m1_R: ArrayLike,
    m2_R: ArrayLike,
    m3_R: ArrayLike,
    E_R: ArrayLike,
    gamma: float,
    passives_L: Optional[ArrayLike],
    conserved_passives_L: Optional[ArrayLike],
    passives_R: Optional[ArrayLike],
    conserved_passives_R: Optional[ArrayLike],
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, Optional[ArrayLike]]:
    c_L = sound_speed(rho_L, P_L, gamma) + xp.abs(v1_L)
    c_R = sound_speed(rho_R, P_R, gamma) + xp.abs(v1_R)
    c_max = xp.maximum(c_L, c_R)

    s_L = xp.minimum(v1_L, v1_R) - c_max
    s_R = xp.maximum(v1_L, v1_R) + c_max

    rc_L = rho_L * (v1_L - s_L)
    rc_R = rho_R * (s_R - v1_R)

    vP_star_denom = avoid_0(rc_L + rc_R, 1e-16)
    v_star = (rc_R * v1_R + rc_L * v1_L + (P_L - P_R)) / vP_star_denom
    P_star = (rc_R * P_L + rc_L * P_R + rc_L * rc_R * (v1_L - v1_R)) / vP_star_denom

    # Star region conservative variables
    r_star_L_denom = avoid_0(s_L - v1_L, 1e-16)
    r_star_R_denom = avoid_0(s_R - v1_R, 1e-16)
    e_star_L_denom = avoid_0(s_L - v_star, 1e-16)
    e_star_R_denom = avoid_0(s_R - v_star, 1e-16)
    r_star_L = rho_L * (s_L - v1_L) / r_star_L_denom
    r_star_R = rho_R * (s_R - v1_R) / r_star_R_denom
    e_star_L = ((s_L - v1_L) * E_L - P_L * v1_L + P_star * v_star) / e_star_L_denom
    e_star_R = ((s_R - v1_R) * E_R - P_R * v1_R + P_star * v_star) / e_star_R_denom

    # Star region conservative variables
    r_gdv = xp.where(
        s_L > 0,
        rho_L,
        xp.where(v_star > 0, r_star_L, xp.where(s_R > 0, r_star_R, rho_R)),
    )
    v_gdv = xp.where(
        s_L > 0, v1_L, xp.where(v_star > 0, v_star, xp.where(s_R > 0, v_star, v1_R))
    )
    P_gdv = xp.where(
        s_L > 0, P_L, xp.where(v_star > 0, P_star, xp.where(s_R > 0, P_star, P_R))
    )
    e_gdv = xp.where(
        s_L > 0, E_L, xp.where(v_star > 0, e_star_L, xp.where(s_R > 0, e_star_R, E_R))
    )

    # Fluxes
    F_rho = r_gdv * v_gdv
    F_m1 = F_rho * v_gdv + P_gdv
    F_E = v_gdv * (e_gdv + P_gdv)
    F_m2 = F_rho * xp.where(v_star > 0, v2_L, v2_R)
    F_m3 = F_rho * xp.where(v_star > 0, v3_L, v3_R)
    if not any(x is None for x in [passives_L, passives_R]):
        F_conserved_passives = F_rho[xp.newaxis, ...] * xp.where(
            v_star > 0, passives_L, passives_R
        )
    else:
        F_conserved_passives = None

    return F_rho, F_m1, F_m2, F_m3, F_E, F_conserved_passives


def hllc(
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
    primitives: bool = True,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, Optional[ArrayLike]]:
    """
    Compute the HLLC fluxes.

    Args:
        rho_L (ArrayLike): Left density.
        v1_L (ArrayLike): Left principal velocity component.
        v2_L (ArrayLike): Left first transverse velocity component.
        v3_L (ArrayLike): Left second transverse velocity component.
        P_L (ArrayLike): Left pressure.
        rho_R (ArrayLike): Right density.
        v1_R (ArrayLike): Right principal velocity component.
        v2_R (ArrayLike): Right first transverse velocity component.
        v3_R (ArrayLike): Right second transverse velocity component.
        P_R (ArrayLike): Right pressure.
        gamma (float): Adiabatic index.
        passives_L (Optional[ArrayLike]): Left passive scalars density
            (rho*passives).
        passives_R (Optional[ArrayLike]): Right passive scalars density
            (rho*passives).
        primitives (bool): Whether the input variables are primitive variables. If
            False, they are considered conservative variables.

    Returns:
        Tuple[ArrayLike, ...]: Fluxes.
        - F_rho (ArrayLike): Density flux.
        - F_m1 (ArrayLike): Momentum flux in the principal direction.
        - F_m2 (ArrayLike): Momentum flux in the first transverse direction.
        - F_m3 (ArrayLike): Momentum flux in the second transverse direction.
        - F_E (ArrayLike): Total energy flux.
        - F_conserved_passives (Optional[ArrayLike]): Passive scalars flux. None if
            `passives` is None.
    """
    return call_riemann_solver(
        _hllc,
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
        passives_L,
        passives_R,
    )
