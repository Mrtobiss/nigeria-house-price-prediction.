"""Streamlit app: estimate the asking price of a residential property in Nigeria.

Run locally with:  streamlit run app.py
"""
import streamlit as st

import predictor

st.set_page_config(page_title="Nigeria House Price Estimator", page_icon="🏠", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def get_bundle():
    return predictor.load_bundle()


st.title("🏠 Nigeria House Price Estimator")
st.write("Describe a residential property to get an estimated **asking price** in Nigerian Naira (₦).")

try:
    bundle = get_bundle()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

meta = bundle["meta"]
options = meta["options"]
ranges = options["ranges"]

st.subheader("Property details")
col_a, col_b = st.columns(2)
with col_a:
    state = st.selectbox("State", sorted(options["locations"]),
                         index=sorted(options["locations"]).index("Lagos") if "Lagos" in options["locations"] else 0)
with col_b:
    town = st.selectbox("Town / area", options["locations"][state])

title = st.selectbox("Property type", options["titles"])

c1, c2, c3, c4 = st.columns(4)
bedrooms = c1.number_input("Bedrooms", ranges["bedrooms"][0], ranges["bedrooms"][1], min(4, ranges["bedrooms"][1]), 1)
bathrooms = c2.number_input("Bathrooms", ranges["bathrooms"][0], ranges["bathrooms"][1], min(4, ranges["bathrooms"][1]), 1)
toilets = c3.number_input("Toilets", ranges["toilets"][0], ranges["toilets"][1], min(5, ranges["toilets"][1]), 1)
parking = c4.number_input("Parking spaces", ranges["parking_space"][0], ranges["parking_space"][1],
                          min(4, ranges["parking_space"][1]), 1)

if st.button("Estimate price", type="primary"):
    try:
        result = predictor.predict(bundle, state=state, town=town, title=title, bedrooms=int(bedrooms),
                                   bathrooms=int(bathrooms), toilets=int(toilets), parking_space=int(parking))
    except ValueError as err:
        st.error(f"Please check the inputs: {err}")
    else:
        st.subheader("Estimate")
        st.metric("Estimated asking price", predictor.format_naira(result.estimate))
        st.caption(f"Based on {result.town_listings:,} comparable listings in {town}.")
        for note in result.notes:
            st.warning(note)

with st.expander("How reliable is this estimate?"):
    m = meta["test_metrics"]
    st.write(
        f"The model ({meta['model']}) was trained on {meta['n_train']:,} property listings and checked on "
        f"{meta['n_test']:,} listings it had never seen. On those, the typical (median) error was about "
        f"**{m['Median error %']:.0f}%**, and **{m['Within 30%']:.0f}%** of estimates were within 30% of the "
        f"actual asking price."
    )
    st.markdown("**Limits to keep in mind**")
    for line in meta["assumptions"]:
        st.markdown(f"- {line}")

st.caption("This tool gives a rough guide based on past listings. It is not a valuation. "
           f"Model trained {meta['created']}.")
