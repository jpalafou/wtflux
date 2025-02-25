import numpy as np
import pytest

from wtflux.hydro import (
    advection_upwinding,
    conservatives_from_primitives,
    fluxes,
    llf,
    primitives_from_conservatives,
    sound_speed,
)


def random_hydro_data(N, n_passives=0):
    rho = np.random.rand(N, N, N)
    vx = np.random.rand(N, N, N)
    vy = np.random.rand(N, N, N)
    vz = np.random.rand(N, N, N)
    P = np.random.rand(N, N, N)
    passives = np.random.rand(n_passives, N, N, N) if n_passives > 0 else None
    return rho, vx, vy, vz, P, passives


def l2_norm(a, b):
    return np.mean(np.square(a - b))


def test_sound_speed():
    """
    Test the sound speed calculation.
    """
    N = 32
    gamma = 5 / 3
    rho, _, _, _, P, _ = random_hydro_data(N)
    sound_speed(rho, P, gamma)


@pytest.mark.parametrize("N", [16, 32, 64])
@pytest.mark.parametrize("gamma", [1.001, 5 / 3, 1.4])
@pytest.mark.parametrize("n_passives", [0, 1, 2, 3])
def test_primitive_conservative_invertability(N, gamma, n_passives):
    """
    Convert primitive variables to conservative variables and back to primitive
    variables.
    """
    HAS_PASSIVES = n_passives > 0

    # define primitive variables
    rho, vx, vy, vz, P, passives = random_hydro_data(N, n_passives)

    # compute conservative variables
    _rho, mx, my, mz, E, _passives = conservatives_from_primitives(
        rho, vx, vy, vz, P, gamma, passives
    )

    # back to primitive variables
    _rho, _vx, _vy, _vz, _P, _passives = primitives_from_conservatives(
        _rho, mx, my, mz, E, gamma, _passives
    )

    # check if the original and the reconstructed primitive variables are the same
    assert l2_norm(rho, _rho) < 1e-15
    assert l2_norm(vx, _vx) < 1e-15
    assert l2_norm(vy, _vy) < 1e-15
    assert l2_norm(vz, _vz) < 1e-15
    assert l2_norm(P, _P) < 1e-15
    if HAS_PASSIVES:
        assert l2_norm(passives, _passives) < 1e-15


@pytest.mark.parametrize("N", [16, 32, 64])
@pytest.mark.parametrize("gamma", [1.001, 5 / 3, 1.4])
@pytest.mark.parametrize("n_passives", [0, 1, 2, 3])
def test_conservative_primitive_invertability(N, gamma, n_passives):
    """
    Convert conservative variables to primitive variables and back to conservative
    variables.
    """
    HAS_PASSIVES = n_passives > 0

    # define conservative variables
    rho, mx, my, mz, E, passives = random_hydro_data(N, n_passives)

    # compute primitive variables
    _rho, _vx, _vy, _vz, _P, _passives = primitives_from_conservatives(
        rho, mx, my, mz, E, gamma, passives
    )

    # back to conservative variables
    _rho, _mx, _my, _mz, _E, _passives = conservatives_from_primitives(
        _rho, _vx, _vy, _vz, _P, gamma, _passives
    )

    # check if the original and the reconstructed conservative variables are the same
    assert l2_norm(rho, _rho) < 1e-15
    assert l2_norm(mx, _mx) < 1e-15
    assert l2_norm(my, _my) < 1e-15
    assert l2_norm(mz, _mz) < 1e-15
    assert l2_norm(E, _E) < 1e-15
    if HAS_PASSIVES:
        assert l2_norm(passives, _passives) < 1e-15


@pytest.mark.parametrize("n_passives", [0, 1, 2, 3])
def test_fluxes(n_passives):
    """
    Test the flux calculation.
    """
    N = 32
    gamma = 5 / 3
    rho, vx, vy, vz, P, passives = random_hydro_data(N, n_passives)
    fluxes(rho, vx, vy, vz, P, gamma, passives)


@pytest.mark.parametrize("precompute_conservatives", [True, False])
@pytest.mark.parametrize("n_passives", [0, 1, 2, 3])
@pytest.mark.parametrize("v", [-1, 1])
def test_advection_upwinding(precompute_conservatives, n_passives, v):
    """
    Test the advection upwinding flux calculation.
    """
    N = 32
    rho_L, vx_L, vy_L, vz_L, _, passives_L = random_hydro_data(N, n_passives)
    rho_R, vx_R, vy_R, vz_R, _, passives_R = random_hydro_data(N, n_passives)
    vx_L[...] = v
    vx_R[...] = v
    F_rho, F_vx, F_vy, F_vz, F_passives = advection_upwinding(
        rho_L,
        vx_L,
        vy_L,
        vz_L,
        rho_R,
        vx_R,
        vy_R,
        vz_R,
        passives_L,
        passives_R,
    )
    assert np.all(F_rho / v == (rho_L if v > 0 else rho_R))
    assert np.all(F_vx / v == (vx_L if v > 0 else vx_R))
    assert np.all(F_vy / v == (vy_L if v > 0 else vy_R))
    assert np.all(F_vz / v == (vz_L if v > 0 else vz_R))
    if n_passives > 0:
        assert np.all(F_passives / v == (passives_L if v > 0 else passives_R))


@pytest.mark.parametrize("precompute_conservatives", [True, False])
@pytest.mark.parametrize("n_passives", [0, 1, 2, 3])
def test_llf(precompute_conservatives, n_passives):
    """
    Test the LLF flux calculation.
    """
    N = 32
    gamma = 5 / 3
    rho_L, vx_L, vy_L, vz_L, P_L, passives_L = random_hydro_data(N, n_passives)
    rho_R, vx_R, vy_R, vz_R, P_R, passives_R = random_hydro_data(N, n_passives)
    if precompute_conservatives:
        _, mx_L, my_L, mz_L, E_L, conserved_passives_L = conservatives_from_primitives(
            rho_L, vx_L, vy_L, vz_L, P_L, gamma, passives_L
        )
        _, mx_R, my_R, mz_R, E_R, conserved_passives_R = conservatives_from_primitives(
            rho_R, vx_R, vy_R, vz_R, P_R, gamma, passives_R
        )
    else:
        _, mx_L, my_L, mz_L, E_L, conserved_passives_L = [None] * 6
        _, mx_R, my_R, mz_R, E_R, conserved_passives_R = [None] * 6
    llf(
        rho_L,
        vx_L,
        vy_L,
        vz_L,
        P_L,
        rho_R,
        vx_R,
        vy_R,
        vz_R,
        P_R,
        gamma,
        passives_L,
        passives_R,
        mx_L,
        my_L,
        mz_L,
        E_L,
        mx_R,
        my_R,
        mz_R,
        E_R,
        conserved_passives_L,
        conserved_passives_R,
    )
