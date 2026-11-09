"""Crop yield estimator based on simple agronomic rules.

This is NOT a machine-learning model trained on a dataset. It starts from the
best yield a crop can reach on a good farm (the "potential yield") and reduces
it for each thing that holds the crop back: water, fertilizer, soil, sowing
date, seed, pests, weeds and weather. All yields are in quintals per hectare
(1 quintal = 100 kg). The numbers are typical Indian values and give a rough
planning estimate, not a guarantee.
"""

from dataclasses import dataclass, field
import math

# potential / typical: quintal per hectare
# water: seasonal water need in mm (rain + irrigation)
# npk: recommended N, P2O5, K2O in kg per hectare
# ph: best soil pH range
# late: yield lost per day of delayed sowing (fraction)
CROPS = {
    "Wheat": dict(potential=55, typical=35, water=(350, 500), npk=(120, 60, 40),
                  ph=(6.0, 7.5), late=0.010, season="Rabi (winter)"),
    "Rice (paddy)": dict(potential=65, typical=28, water=(900, 1500), npk=(120, 60, 40),
                         ph=(5.5, 7.0), late=0.006, season="Kharif (monsoon)"),
    "Maize": dict(potential=75, typical=30, water=(500, 800), npk=(120, 60, 40),
                  ph=(5.8, 7.0), late=0.007, season="Kharif / Rabi"),
    "Cotton (seed cotton)": dict(potential=35, typical=20, water=(700, 1100), npk=(100, 50, 50),
                                 ph=(6.0, 8.0), late=0.006, season="Kharif (monsoon)"),
    "Sugarcane": dict(potential=1100, typical=840, water=(1500, 2500), npk=(150, 60, 60),
                      ph=(6.0, 7.5), late=0.004, season="Annual"),
    "Soybean": dict(potential=30, typical=11, water=(450, 700), npk=(25, 60, 40),
                    ph=(6.0, 7.0), late=0.006, season="Kharif (monsoon)"),
    "Potato": dict(potential=400, typical=230, water=(400, 600), npk=(150, 80, 100),
                   ph=(5.0, 6.5), late=0.010, season="Rabi (winter)"),
    "Mustard": dict(potential=28, typical=14, water=(250, 400), npk=(80, 40, 40),
                    ph=(6.0, 7.5), late=0.010, season="Rabi (winter)"),
    "Gram (chickpea)": dict(potential=25, typical=11, water=(250, 400), npk=(20, 50, 20),
                            ph=(6.0, 7.5), late=0.010, season="Rabi (winter)"),
}

SOILS = ["Alluvial (Gangetic plain)", "Loam", "Clay", "Black (regur)", "Red", "Sandy"]

# how well each soil suits each crop (1.0 = best)
SOIL_FIT = {
    "Wheat":                {"Loam": 1.00, "Alluvial (Gangetic plain)": 1.00, "Clay": 0.92, "Black (regur)": 0.95, "Red": 0.85, "Sandy": 0.75},
    "Rice (paddy)":         {"Clay": 1.00, "Alluvial (Gangetic plain)": 0.97, "Loam": 0.95, "Black (regur)": 0.90, "Red": 0.80, "Sandy": 0.60},
    "Maize":                {"Loam": 1.00, "Alluvial (Gangetic plain)": 0.98, "Red": 0.88, "Black (regur)": 0.90, "Clay": 0.85, "Sandy": 0.80},
    "Cotton (seed cotton)": {"Black (regur)": 1.00, "Alluvial (Gangetic plain)": 0.93, "Loam": 0.92, "Red": 0.80, "Clay": 0.85, "Sandy": 0.70},
    "Sugarcane":            {"Loam": 1.00, "Alluvial (Gangetic plain)": 0.98, "Clay": 0.95, "Black (regur)": 0.93, "Red": 0.80, "Sandy": 0.70},
    "Soybean":              {"Black (regur)": 1.00, "Loam": 0.97, "Alluvial (Gangetic plain)": 0.93, "Red": 0.85, "Clay": 0.85, "Sandy": 0.70},
    "Potato":               {"Loam": 1.00, "Alluvial (Gangetic plain)": 0.97, "Sandy": 0.88, "Red": 0.85, "Black (regur)": 0.75, "Clay": 0.70},
    "Mustard":              {"Alluvial (Gangetic plain)": 1.00, "Loam": 0.98, "Sandy": 0.85, "Black (regur)": 0.90, "Red": 0.85, "Clay": 0.85},
    "Gram (chickpea)":      {"Black (regur)": 1.00, "Loam": 0.98, "Alluvial (Gangetic plain)": 0.95, "Red": 0.88, "Sandy": 0.85, "Clay": 0.80},
}

SEED = {
    "Improved / hybrid seed": 1.00,
    "Certified seed of a regular variety": 0.93,
    "Own saved / local seed": 0.82,
}
PEST = {"None seen": 1.00, "Light": 0.94, "Moderate": 0.82, "Severe": 0.62}
WEEDS = {"Controlled well": 1.00, "Partly controlled": 0.92, "Poorly controlled": 0.80}
WEATHER = {"None": 1.00, "Mild (a few hot/cold/dry days)": 0.92, "Severe (frost, heat wave, flood or long drought)": 0.75}

IRRIGATION_MM = 50  # water in one irrigation, a typical value
FYM_EFFECTIVE = (2.0, 1.0, 2.0)  # kg N, P2O5, K2O available per tonne of farmyard manure in year one


@dataclass
class FarmInput:
    crop: str
    area: float                 # hectares
    soil: str
    ph: float | None            # None when the farmer does not know it
    rainfall_mm: float          # rain during the crop season
    irrigations: int
    n_kg: float                 # applied per hectare
    p_kg: float
    k_kg: float
    manure_t: float             # farmyard manure, tonnes per hectare
    sowing_delay_days: int
    seed: str
    pests: str
    weeds: str
    weather: str


@dataclass
class Result:
    yield_per_ha: float
    low_per_ha: float
    high_per_ha: float
    total: float
    total_low: float
    total_high: float
    potential: float
    typical: float
    factors: dict = field(default_factory=dict)  # name -> 0..1


def _nutrient_response(ratio: float) -> float:
    """0..1 response to a nutrient. Rises quickly, flattens near the recommended dose,
    and drops slightly when far above it."""
    if ratio <= 1:
        return (1 - math.exp(-2.5 * ratio)) / (1 - math.exp(-2.5))
    return max(0.85, 1 - 0.10 * (ratio - 1))


def fertilizer_factor(crop: str, n: float, p: float, k: float, manure_t: float) -> float:
    rec_n, rec_p, rec_k = CROPS[crop]["npk"]
    n += manure_t * FYM_EFFECTIVE[0]
    p += manure_t * FYM_EFFECTIVE[1]
    k += manure_t * FYM_EFFECTIVE[2]
    resp = (0.5 * _nutrient_response(n / rec_n)
            + 0.25 * _nutrient_response(p / rec_p)
            + 0.25 * _nutrient_response(k / rec_k))
    return 0.55 + 0.45 * resp  # unfertilized crops still give about 55%


def water_factor(crop: str, rainfall_mm: float, irrigations: int) -> float:
    lo, hi = CROPS[crop]["water"]
    total = rainfall_mm + irrigations * IRRIGATION_MM
    if total < lo:
        return 0.30 + 0.70 * (total / lo)
    if total <= hi:
        return 1.0
    return max(0.65, 1 - 0.35 * min(1.0, total / hi - 1))  # waterlogging


def ph_factor(crop: str, ph: float | None) -> float:
    if ph is None:
        return 0.97
    lo, hi = CROPS[crop]["ph"]
    gap = lo - ph if ph < lo else (ph - hi if ph > hi else 0.0)
    return max(0.60, 1 - 0.12 * gap)


def sowing_factor(crop: str, delay_days: int) -> float:
    return 1 - min(0.40, CROPS[crop]["late"] * max(0, delay_days))


def predict(f: FarmInput) -> Result:
    c = CROPS[f.crop]
    factors = {
        "Water": water_factor(f.crop, f.rainfall_mm, f.irrigations),
        "Fertilizer": fertilizer_factor(f.crop, f.n_kg, f.p_kg, f.k_kg, f.manure_t),
        "Soil type": SOIL_FIT[f.crop][f.soil],
        "Soil pH": ph_factor(f.crop, f.ph),
        "Sowing time": sowing_factor(f.crop, f.sowing_delay_days),
        "Seed": SEED[f.seed],
        "Pests and disease": PEST[f.pests],
        "Weeds": WEEDS[f.weeds],
        "Weather": WEATHER[f.weather],
    }
    # Even a very poor crop gives something; stacking every worst case would be unrealistic.
    score = max(0.12, math.prod(factors.values()))
    per_ha = c["potential"] * score
    low, high = per_ha * 0.88, per_ha * 1.12
    return Result(per_ha, low, high, per_ha * f.area, low * f.area, high * f.area,
                  c["potential"], c["typical"], factors)


def advice(f: FarmInput, r: Result) -> list[str]:
    """Plain-language tips for the factors that cost the most yield."""
    c = CROPS[f.crop]
    lo_w, hi_w = c["water"]
    rec_n, rec_p, rec_k = c["npk"]
    tips = {
        "Water": (f"Your crop gets about {f.rainfall_mm + f.irrigations * IRRIGATION_MM:.0f} mm of water in the season; "
                  f"{f.crop} does best with {lo_w}-{hi_w} mm. "
                  + ("Add irrigations at the key stages (crown root, flowering, grain filling)." if f.rainfall_mm + f.irrigations * IRRIGATION_MM < lo_w
                     else "Improve drainage so water does not stand in the field.")),
        "Fertilizer": (f"Recommended dose is about N {rec_n}, P2O5 {rec_p}, K2O {rec_k} kg per hectare. "
                       "Get a soil test and apply farmyard manure or compost along with it."),
        "Soil type": "Add compost or green manure to improve how the soil holds water and nutrients.",
        "Soil pH": f"Best pH for {f.crop} is {c['ph'][0]}-{c['ph'][1]}. Use gypsum for high pH or lime for low pH, as the soil test advises.",
        "Sowing time": "Sow on time. Every day of delay costs yield, so keep seed, labour and equipment ready before the sowing window.",
        "Seed": "Switch to certified seed of an improved variety from a trusted dealer or the Krishi Vigyan Kendra.",
        "Pests and disease": "Scout the field every week and spray only when needed. Your local KVK can identify the pest and the right treatment.",
        "Weeds": "Weed within the first 30-40 days, when the crop is most sensitive to competition.",
        "Weather": "Consider crop insurance (PMFBY) and weather advisories from IMD / Kisan Suvidha to prepare for extreme weather.",
    }
    worst = sorted(r.factors.items(), key=lambda kv: kv[1])
    return [f"{name} (about {round((1 - v) * 100)}% yield loss): {tips[name]}"
            for name, v in worst if v < 0.95][:4]


if __name__ == "__main__":
    demo = FarmInput("Wheat", 1.0, "Alluvial (Gangetic plain)", 7.2, 60, 4, 100, 50, 30, 5,
                     10, "Certified seed of a regular variety", "Light", "Partly controlled",
                     "None")
    res = predict(demo)
    print(f"{res.yield_per_ha:.1f} q/ha  (range {res.low_per_ha:.1f}-{res.high_per_ha:.1f}), "
          f"typical India {res.typical} q/ha")
    for k, v in res.factors.items():
        print(f"  {k:18s} {v:.2f}")
