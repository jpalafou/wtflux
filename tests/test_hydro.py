import pytest

from wtflux.backend import xp
from wtflux.hydro import (
    conservatives_from_primitives,
    fluxes,
    llf,
    primitives_from_conservatives,
    sound_speed,
)


def random_hydro_data(N, n_passives=0):
    rho = xp.random.rand(N, N, N)
    vx = xp.random.rand(N, N, N)
    vy = xp.random.rand(N, N, N)
    vz = xp.random.rand(N, N, N)
    P = xp.random.rand(N, N, N)
    passives = xp.random.rand(n_passives, N, N, N) if n_passives > 0 else None
    return rho, vx, vy, vz, P, passives


def l2_norm(a, b):
    return xp.mean(xp.square(a - b))


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


@pytest.mark.parametrize("N", [16, 32, 64])
@pytest.mark.parametrize("gamma", [1.001, 5 / 3, 1.4])
def test_primitives_from_conservatives_passives(N, gamma):
    """
    Test that passive variables don't change the result of converting conservatives to
    primitives.
    """
    rho, mx, my, mz, E, passives = random_hydro_data(N, 1)
    u = xp.array([rho, mx, my, mz, E])
    u_with_passives = xp.concatenate([xp.array([rho, mx, my, mz, E]), passives])
    w = primitives_from_conservatives(u[0], u[1], u[2], u[3], u[4], gamma)
    w_with_passives = primitives_from_conservatives(
        u_with_passives[0],
        u_with_passives[1],
        u_with_passives[2],
        u_with_passives[3],
        u_with_passives[4],
        gamma,
        u_with_passives[5:],
    )
    for i in range(5):
        assert xp.all(w[i] == w_with_passives[i])


@pytest.mark.parametrize("N", [16, 32, 64])
@pytest.mark.parametrize("gamma", [1.001, 5 / 3, 1.4])
def test_conservatives_from_primitives_passives(N, gamma):
    """
    Test that passive variables don't change the result of converting primitives to
    conservatives.
    """
    rho, vx, vy, vz, P, passives = random_hydro_data(N, 1)
    w = xp.array([rho, vx, vy, vz, P])
    w_with_passives = xp.concatenate([xp.array([rho, vx, vy, vz, P]), passives])
    u = conservatives_from_primitives(w[0], w[1], w[2], w[3], w[4], gamma)
    u_with_passives = conservatives_from_primitives(
        w_with_passives[0],
        w_with_passives[1],
        w_with_passives[2],
        w_with_passives[3],
        w_with_passives[4],
        gamma,
        w_with_passives[5:],
    )
    for i in range(5):
        assert xp.all(u[i] == u_with_passives[i])


@pytest.mark.parametrize("n_passives", [0, 1, 2, 3])
def test_fluxes(n_passives):
    """
    Test the flux calculation.
    """
    N = 32
    gamma = 5 / 3
    rho, vx, vy, vz, P, passives = random_hydro_data(N, n_passives)
    fluxes(rho, vx, vy, vz, P, gamma, passives)


@pytest.mark.parametrize("primitive_inputs", [True, False])
@pytest.mark.parametrize("n_passives", [0, 1, 2, 3])
def test_llf(primitive_inputs, n_passives):
    """
    Test the LLF flux calculation.
    """
    N = 32
    gamma = 5 / 3
    rho_L, vx_L, vy_L, vz_L, P_L, passives_L = random_hydro_data(N, n_passives)
    rho_R, vx_R, vy_R, vz_R, P_R, passives_R = random_hydro_data(N, n_passives)
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
        primitive_inputs,
    )


@pytest.mark.parametrize("primitive_inputs", [True, False])
def test_llf_with_passives(primitive_inputs):
    """
    Test that the the LLF flux calculation is unchanged by passive variables.
    """
    N = 32
    gamma = 5 / 3
    rho_L, vx_L, vy_L, vz_L, P_L, passives_L = random_hydro_data(N, 1)
    rho_R, vx_R, vy_R, vz_R, P_R, passives_R = random_hydro_data(N, 1)

    F = llf(
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
        primitives=primitive_inputs,
    )
    F_with_passives = llf(
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
        primitives=primitive_inputs,
    )
    for i in range(5):
        assert xp.all(F[i] == F_with_passives[i])
