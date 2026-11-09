"""Crop Yield Predictor: a simple form for farmers. Run with: streamlit run app.py"""

import pandas as pd
import streamlit as st

from predictor import (CROPS, SOILS, SEED, PEST, WEEDS, WEATHER, FarmInput,
                       advice, predict)

# Rough market prices in Rs per quintal. Farmers should replace these with their local mandi rate.
DEFAULT_PRICE = {
    "Wheat": 2400, "Rice (paddy)": 2300, "Maize": 2200, "Cotton (seed cotton)": 7000,
    "Sugarcane": 370, "Soybean": 4800, "Potato": 1500, "Mustard": 5600, "Gram (chickpea)": 5600,
}
ACRE_IN_HECTARE = 0.4047

st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="centered")

st.title("🌾 Crop Yield Predictor")
st.caption("Answer a few questions about your field to get an estimate of your harvest. "
           "This is a planning estimate based on standard farming rules, not a guarantee.")

with st.form("farm"):
    st.subheader("1. Crop and land")
    c1, c2, c3 = st.columns([2, 1, 1])
    crop = c1.selectbox("Crop", list(CROPS), help="The crop you are growing this season.")
    area = c2.number_input("Land area", min_value=0.1, value=1.0, step=0.1)
    unit = c3.selectbox("Unit", ["Acre", "Hectare"])

    st.subheader("2. Soil")
    s1, s2 = st.columns(2)
    soil = s1.selectbox("Soil type", SOILS)
    ph_known = s2.checkbox("I know my soil pH (from a soil test)", value=False)
    ph = s2.number_input("Soil pH", min_value=4.0, max_value=9.5, value=7.0, step=0.1,
                         disabled=not ph_known)

    st.subheader("3. Water")
    w1, w2 = st.columns(2)
    rainfall = w1.number_input("Rainfall during the crop season (mm)", min_value=0, max_value=4000,
                               value=100, step=10,
                               help="Total rain from sowing to harvest. Check your local weather station or IMD.")
    irrigations = w2.number_input("Number of irrigations", min_value=0, max_value=30, value=4,
                                  help="How many times you watered the field.")

    st.subheader("4. Fertilizer (per hectare)")
    st.caption("Enter kg of nutrient per hectare. 1 bag of urea (50 kg) has about 23 kg nitrogen; "
               "1 bag of DAP (50 kg) has about 23 kg nitrogen and 23 kg phosphorus.")
    f1, f2, f3, f4 = st.columns(4)
    n_kg = f1.number_input("Nitrogen (N)", min_value=0, max_value=400, value=100, step=10)
    p_kg = f2.number_input("Phosphorus (P2O5)", min_value=0, max_value=200, value=50, step=10)
    k_kg = f3.number_input("Potash (K2O)", min_value=0, max_value=200, value=30, step=10)
    manure = f4.number_input("Farmyard manure (tonnes)", min_value=0.0, max_value=30.0, value=5.0, step=0.5)

    st.subheader("5. Farming practices")
    p1, p2 = st.columns(2)
    delay = p1.slider("Days sowing was late compared to the best time", 0, 45, 0)
    seed = p2.selectbox("Seed used", list(SEED))
    pests = p1.selectbox("Pest and disease attack", list(PEST))
    weeds = p2.selectbox("Weeds", list(WEEDS))
    weather = p1.selectbox("Bad weather this season (frost, heat wave, flood, drought)", list(WEATHER))
    price = p2.number_input("Selling price (Rs per quintal)", min_value=0, value=DEFAULT_PRICE[crop], step=50,
                            help="Change this to the rate in your local mandi. The default is only a rough guide.")

    submitted = st.form_submit_button("Predict my yield", type="primary", use_container_width=True)

if submitted:
    area_ha = area * ACRE_IN_HECTARE if unit == "Acre" else area
    farm = FarmInput(crop=crop, area=area_ha, soil=soil, ph=ph if ph_known else None,
                     rainfall_mm=rainfall, irrigations=irrigations, n_kg=n_kg, p_kg=p_kg, k_kg=k_kg,
                     manure_t=manure, sowing_delay_days=delay, seed=seed, pests=pests,
                     weeds=weeds, weather=weather)
    res = predict(farm)

    st.divider()
    st.header("Your estimated harvest")
    unit_area = "acre" if unit == "Acre" else "hectare"
    per_unit = res.yield_per_ha * ACRE_IN_HECTARE if unit == "Acre" else res.yield_per_ha

    m1, m2, m3 = st.columns(3)
    m1.metric("Total harvest", f"{res.total:,.1f} quintal",
              help=f"Likely range {res.total_low:,.1f} to {res.total_high:,.1f} quintal")
    m2.metric(f"Per {unit_area}", f"{per_unit:,.1f} quintal")
    m3.metric("Estimated income", f"Rs {res.total * price:,.0f}",
              help="Harvest multiplied by your selling price. This is before deducting costs.")
    st.caption(f"Likely range for the whole field: {res.total_low:,.1f} to {res.total_high:,.1f} quintal. "
               f"(1 quintal = 100 kg)")

    diff = (res.yield_per_ha / res.typical - 1) * 100
    if diff >= 5:
        st.success(f"Your estimate is {diff:.0f}% above the typical Indian yield for {crop} "
                   f"({res.typical} quintal per hectare).")
    elif diff <= -5:
        st.warning(f"Your estimate is {abs(diff):.0f}% below the typical Indian yield for {crop} "
                   f"({res.typical} quintal per hectare).")
    else:
        st.info(f"Your estimate is close to the typical Indian yield for {crop} "
                f"({res.typical} quintal per hectare).")

    st.subheader("What is holding your yield back")
    st.caption("100% means this factor is not reducing your yield at all.")
    df = pd.DataFrame({"Factor": list(res.factors), "Score (%)": [round(v * 100) for v in res.factors.values()]})
    st.bar_chart(df, x="Factor", y="Score (%)", horizontal=True)

    tips = advice(farm, res)
    if tips:
        st.subheader("How to improve")
        for t in tips:
            st.markdown(f"- {t}")
    else:
        st.success("Your practices look good. Keep it up.")

    st.caption("Estimates use typical values for Indian conditions. Your actual harvest also depends on "
               "the variety, local climate and field management. Talk to your nearest Krishi Vigyan Kendra for advice.")
