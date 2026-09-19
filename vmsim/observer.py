"""Observing sites and the geocentric -> topocentric transformation."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .frames import Vec3
from .timescale import gmst_deg

EARTH_EQUATORIAL_RADIUS_KM = 6378.140
FLATTENING_FACTOR = 0.99664719          # b/a for the IAU 1976 ellipsoid


@dataclass(frozen=True)
class Site:
    name: str
    latitude: float          # degrees, north positive
    longitude: float         # degrees, EAST positive
    elevation_m: float = 0.0
    region: str = ""

    def geocentric_terms(self) -> tuple[float, float]:
        """(rho*sin(phi'), rho*cos(phi')) in Earth equatorial radii (Meeus 11.1)."""
        phi = math.radians(self.latitude)
        u = math.atan(FLATTENING_FACTOR * math.tan(phi))
        h = self.elevation_m / (EARTH_EQUATORIAL_RADIUS_KM * 1000.0)
        rho_sin = FLATTENING_FACTOR * math.sin(u) + h * math.sin(phi)
        rho_cos = math.cos(u) + h * math.cos(phi)
        return rho_sin, rho_cos

    def position_equatorial_km(self, local_apparent_sidereal_deg: float) -> Vec3:
        """Observer's offset from the Earth's centre, equatorial-of-date, in km."""
        rho_sin, rho_cos = self.geocentric_terms()
        lst = math.radians(local_apparent_sidereal_deg)
        r = EARTH_EQUATORIAL_RADIUS_KM
        return (r * rho_cos * math.cos(lst), r * rho_cos * math.sin(lst), r * rho_sin)

    def local_sidereal_deg(self, jd_ut: float, equation_of_equinoxes_deg: float = 0.0) -> float:
        return (gmst_deg(jd_ut) + equation_of_equinoxes_deg + self.longitude) % 360.0


# Sites chosen to span the September 14 2026 occultation: inside the track
# (Asia / Africa / Europe) and outside it, so the difference is visible.
SITES: tuple[Site, ...] = (
    Site("Chennai",      13.0827,  80.2707,    6, "India"),
    Site("Bengaluru",    12.9716,  77.5946,  920, "India"),
    Site("New Delhi",    28.6139,  77.2090,  216, "India"),
    Site("Mumbai",       19.0760,  72.8777,   14, "India"),
    Site("Kolkata",      22.5726,  88.3639,    9, "India"),
    Site("Colombo",       6.9271,  79.8612,    5, "Sri Lanka"),
    Site("Dhaka",        23.8103,  90.4125,    4, "Bangladesh"),
    Site("Kathmandu",    27.7172,  85.3240, 1400, "Nepal"),
    Site("Karachi",      24.8607,  67.0011,   10, "Pakistan"),
    Site("Dubai",        25.2048,  55.2708,    5, "UAE"),
    Site("Riyadh",       24.7136,  46.6753,  612, "Saudi Arabia"),
    Site("Cairo",        30.0444,  31.2357,   23, "Egypt"),
    Site("Nairobi",      -1.2921,  36.8219, 1795, "Kenya"),
    Site("Addis Ababa",   9.0320,  38.7469, 2355, "Ethiopia"),
    Site("Istanbul",     41.0082,  28.9784,   39, "Turkiye"),
    Site("Athens",       37.9838,  23.7275,   70, "Greece"),
    Site("Rome",         41.9028,  12.4964,   21, "Italy"),
    Site("Madrid",       40.4168,  -3.7038,  667, "Spain"),
    Site("London",       51.5072,  -0.1276,   11, "UK"),
    Site("Moscow",       55.7558,  37.6173,  156, "Russia"),
    Site("Beijing",      39.9042, 116.4074,   44, "China"),
    Site("Singapore",     1.3521, 103.8198,   15, "Singapore"),
    Site("Perth",       -31.9523, 115.8613,   30, "Australia"),
    Site("Johannesburg",-26.2041,  28.0473, 1753, "South Africa"),
    Site("New York",     40.7128, -74.0060,   10, "USA"),
    Site("Sao Paulo",   -23.5505, -46.6333,  760, "Brazil"),
)

SITES_BY_NAME = {s.name.lower(): s for s in SITES}


def find_site(name: str) -> Site:
    key = name.strip().lower()
    if key in SITES_BY_NAME:
        return SITES_BY_NAME[key]
    for k, v in SITES_BY_NAME.items():
        if k.startswith(key):
            return v
    raise KeyError(f"unknown site {name!r}; known: {', '.join(s.name for s in SITES)}")
