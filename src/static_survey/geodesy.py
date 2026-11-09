"""GDA2020 / MGA2020 geometry on the GRS80 ellipsoid.

Grid coordinates (MGA zone 56 by default) and AHD heights are turned into
Earth-centred XYZ, so every observation can be modelled in true 3D space
rather than on the projection.
"""

from __future__ import annotations

import numpy as np
from pyproj import Transformer

A = 6378137.0
F = 1 / 298.257222101
E2 = F * (2 - F)
RHO = 180.0 / np.pi * 3600.0  # arc-seconds per radian
R_MEAN = 6371000.0

_transformers: dict[int, Transformer] = {}


def _grid_to_geog(epsg: int) -> Transformer:
    if epsg not in _transformers:
        _transformers[epsg] = Transformer.from_crs(epsg, 7844, always_xy=True)  # MGA2020 -> GDA2020 lon/lat
    return _transformers[epsg]


def grid_to_latlon(e, n, epsg: int = 7856):
    lon, lat = _grid_to_geog(epsg).transform(np.asarray(e, float), np.asarray(n, float))
    return np.radians(lat), np.radians(lon)


def latlon_to_grid(lat, lon, epsg: int = 7856):
    lon_d, lat_d = np.degrees(lon), np.degrees(lat)
    e, n = _grid_to_geog(epsg).transform(lon_d, lat_d, direction="INVERSE")
    return np.asarray(e), np.asarray(n)


def geodetic_to_ecef(lat, lon, h):
    lat, lon, h = (np.asarray(x, float) for x in (lat, lon, h))
    sl, cl = np.sin(lat), np.cos(lat)
    nu = A / np.sqrt(1 - E2 * sl * sl)
    x = (nu + h) * cl * np.cos(lon)
    y = (nu + h) * cl * np.sin(lon)
    z = (nu * (1 - E2) + h) * sl
    return np.stack([x, y, z], axis=-1)


def ecef_to_geodetic(xyz):
    xyz = np.atleast_2d(np.asarray(xyz, float))
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - E2))
    for _ in range(8):
        nu = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - nu
        lat = np.arctan2(z, p * (1 - E2 * nu / (nu + h)))
    nu = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - nu
    return lat, lon, h


def enu_matrix(lat, lon):
    """Rotation(s) taking an ECEF vector to local east, north, up."""
    lat, lon = np.atleast_1d(lat), np.atleast_1d(lon)
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    r = np.zeros((lat.size, 3, 3))
    r[:, 0] = np.stack([-so, co, np.zeros_like(so)], -1)
    r[:, 1] = np.stack([-sl * co, -sl * so, cl], -1)
    r[:, 2] = np.stack([cl * co, cl * so, sl], -1)
    return r


def grid_point_scale(e, n, epsg: int = 7856):
    """Point scale factor and grid convergence (radians) from pyproj."""
    from pyproj import Proj

    proj = Proj(f"EPSG:{epsg}")
    lat, lon = grid_to_latlon(e, n, epsg)
    fac = proj.get_factors(np.degrees(lon), np.degrees(lat))
    return np.asarray(fac.meridional_scale), np.radians(np.asarray(fac.meridian_convergence))


def grid_ecef(e, n, h_ellipsoidal, epsg: int = 7856, step: float = 0.5):
    """ECEF positions and the 3x3 Jacobian d(XYZ)/d(E, N, h) for each point."""
    e, n, h = (np.asarray(x, float) for x in (e, n, h_ellipsoidal))
    lat, lon = grid_to_latlon(e, n, epsg)
    xyz = geodetic_to_ecef(lat, lon, h)
    jac = np.zeros((e.size, 3, 3))
    for k, (de, dn) in enumerate(((step, 0.0), (0.0, step))):
        lp, op = grid_to_latlon(e + de, n + dn, epsg)
        lm, om = grid_to_latlon(e - de, n - dn, epsg)
        jac[:, :, k] = (geodetic_to_ecef(lp, op, h) - geodetic_to_ecef(lm, om, h)) / (2 * step)
    jac[:, :, 2] = enu_matrix(lat, lon)[:, 2]  # d(XYZ)/dh is the ellipsoid normal
    return xyz, jac, lat, lon
