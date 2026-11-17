"""E1-E4 physical checks (spec section 3), each with a real-data gate."""
from metacf.v2.physics.e1_resolution import E1Resolution
from metacf.v2.physics.e2_season import E2Season
from metacf.v2.physics.e3_sun import E3Sun
from metacf.v2.physics.e4_year import E4Year

ALL = [E1Resolution, E2Season, E3Sun, E4Year]
