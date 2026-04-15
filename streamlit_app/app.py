# =============================================================================
# app.py — GN Food Estimator  |  Streamlit app
# =============================================================================
# Run locally:
#   pip install streamlit numpy pillow opencv-python-headless
#   streamlit run app.py
#
# STUB_MODE is controlled by config.STUB_MODE (True = no model files needed).
# Set it to False once .pth files are placed in models/.
# =============================================================================

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import streamlit as st
from PIL import Image as PILImage
import cv2
import pandas as pd

import config as cfg
from cv_tasks.aruco import draw_aruco_overlay, ArUcoNotFoundError
from cv_tasks.preprocessor import to_bgr
from pipeline import (
    stage1_capture,
    stage2_detection,
    stage3_geometry,
    stage4_fill_level,
    stage5_volume,
    stage6_food_volume,
    stage7_food_id,
    stage8_mass,
    stage9_emissions,
    stage10_11_compare 
)
from pipeline.stage2_detection import ContainerNotFoundError


# =============================================================================
# Page config
# =============================================================================

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lilita+One&family=Open+Sans:wght@400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Open Sans', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Lilita One', sans-serif;
        font-weight: 400;
    }

    /* Sidebar header */
    .css-1lcbmhc {
        font-family: 'Lilita One', sans-serif;
    }

    /* Button styling */
    .stButton>button {
        font-family: 'Open Sans', sans-serif;
        font-weight: 600;
        background-color: #FF6B6B;
        color: white;
        border-radius: 4px;
        border: none;
    }

    /* Success/error messages */
    .stAlert {
        border-radius: 4px;
        font-family: 'Open Sans', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Call this at the start of your app
inject_custom_css()


st.set_page_config(
    page_title = "FBG - GN Food Estimator",
    page_icon = "🍱",
    layout = "centered",
    initial_sidebar_state = "expanded",
    menu_items = {
        'Get Help': 'https://www.foodbegood.app/',
        'Report a bug': "https://www.linkedin.com/company/food-be-good/",
        'About': "# Food Sharing for Visitors and Canteens. *Full Bellies, NOT Full Bins.*"
    }
)

FBG_LOGO = cfg.ROOT / "images" / "fbg-logo.png"
st.logo(FBG_LOGO, size = "large", link = "https://www.foodbegood.app/")

# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.subheader("Canteen summary")
    st.title("**Berlin Campus Canteen**")
    
    st.markdown("Pickup time" \
    "**14:30 - 14:35**" \
    "" \
    "FBG contact person" \
    "**Vanessa Klein**" \
    "" \
    "Agreed users" \
    "**420**")


    st.divider()

    st.subheader("Mode")
    if cfg.STUB_MODE:
        st.info(
            "**Stub mode** — no model files needed.\n\n"
            "Set `STUB_MODE = False` in `config.py` once `.pth` files "
            "are placed in `models/`.",
            icon = "🔧"
        )
    else:
        st.success("**Live mode** — using trained models.", icon = "✅")

    st.divider()

    #st.subheader("Thresholds")
    #score_thresh = st.slider(
    #    "Detection confidence threshold",
    #    min_value = 0.10,
    #    max_value = 0.90,
    #    value = cfg.MODEL1_SCORE_THRESH,
    #    step = 0.05,
    #    help = "Boxes below this score are discarded."
    #)
    #cfg.MODEL1_SCORE_THRESH = score_thresh

    #conf_thresh = st.slider(
    #    "Fill level confidence threshold",
    #    min_value = 0.40,
    #    max_value = 0.95,
    #    value = cfg.CONFIDENCE_THRESHOLD,
    #    step = 0.05,
    #    help = "Below this confidence, you will be asked to retake the photo."
    #)
    #cfg.CONFIDENCE_THRESHOLD = conf_thresh

    st.divider()
    st.caption(cfg.CO2_DISCLAIMER)


# =============================================================================
# Helpers
# =============================================================================

def to_pil(arr: np.ndarray) -> PILImage.Image:
    """
    convert numpy RGB → PIL for Streamlit display
    """
    return PILImage.fromarray(arr.astype(np.uint8))


def draw_detections(
        image_rgb: np.ndarray,
        containers: list) -> np.ndarray:
    """
    Draw bounding boxes and labels on a copy of the image.
    """
    overlay = image_rgb.copy()
    colours = {
        "large container": (232, 89, 60),   # coral
        "small container": (29, 158, 117)  # teal
    }

    for c in containers:
        x0, y0, x1, y1 = [int(v) for v in c.box]
        colour = colours.get(c.label, (150, 150, 150))
        cv2.rectangle(overlay, (x0, y0), (x1, y1), colour, 2)
        txt = f"{c.label} {c.score:.2f}"
        cv2.putText(
            overlay,
            txt,
            (x0 + 4, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2
        )
    return overlay


def run_pipeline_pass1(source_bytes: bytes) -> dict | None:
    """
    Run Stages I-VII on one photo.
    Stops after Stage VII so the app can show food identification results
    and collect manual overrides before continuing to Stages VIII-IX.
    Returns a partial results dict, or None on fatal error.
    """
    try:
        s1 = stage1_capture.run(source_bytes)
    except ArUcoNotFoundError as e:
        st.error(f"**ArUco marker not detected.** {e}")
        st.warning(
            "Ensure the printed marker (ID 0, DICT_4X4_50) is fully "
            "visible and retake the photo."
        )
        return None

    try:
        s2 = stage2_detection.run(s1)
    except ContainerNotFoundError as e:
        st.error(f"**No container detected.** {e}")
        return None

    s3 = stage3_geometry.run(s2)
    s4 = stage4_fill_level.run(s3)
    s5 = stage5_volume.run(s4)
    s6 = stage6_food_volume.run(s5)
    crops = [f.crop_rgb for f in s4.fills]
    s7 = stage7_food_id.run(s6, crops)

    return dict(s1=s1, s2=s2, s3=s3, s4=s4, s5=s5, s6=s6, s7=s7)


def run_pipeline_pass2(partial: dict, overrides: dict) -> dict:
    """
    Run Stages VIII-IX, applying any user overrides to Stage VII first.

    Parameters
    ----------
    partial   : dict returned by run_pipeline_pass1
    overrides : {container_index: food_type_string} — may be empty
    """
    s7 = partial["s7"]
    if overrides:
        s7 = stage7_food_id.apply_overrides(s7, overrides)
    s8 = stage8_mass.run(s7)
    s9 = stage9_emissions.run(s8)
    return {**partial, "s7": s7, "s8": s8, "s9": s9}


def show_photo_results(results: dict, photo_label: str):
    """
    Display the full per-photo results in two columns.
    """
    s1, s2, s3, s4, s9 = (results["s1"], results["s2"], results["s3"], results["s4"], results["s9"])
 
    col_img, col_data = st.columns([1, 1], gap = "large", border = True)
 
    with col_img:
        st.markdown("##### Detection")
        det = draw_detections(s1.image_rgb, s2.containers)
        bgr = draw_aruco_overlay(to_bgr(det), s1.aruco)
        st.image(
            to_pil(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
            width = "stretch",
            caption = "Detected containers + ArUco marker"
        )

        st.markdown("##### Rectified image")
        st.image(
            to_pil(s3.rectified_rgb),
            width = "stretch",
            caption = "After perspective correction (Stage III)"
        )
 
    with col_data:
        st.markdown("##### Container results")
        for i, em in enumerate(s9.emissions):
            food_badge = (
                f":green[{em.food_type}]"
                if em.food_type != "unknown"
                else ":orange[unknown]"
            )
            fill_colour = {
                "empty": "gray",
                "low": "blue",
                "medium": "orange",
                "high": "red",
                "full": "red"
            }.get(em.fill_label, "gray")
 
            with st.expander(
                f"Container {i+1}  |  {em.container_label}  "
                f"|  food: {food_badge}  "
                f"|  fill: :{fill_colour}[{em.fill_label}]",
                expanded = True
            ):
                c1, c2 = st.columns([1, 2])
                c1.image(
                    to_pil(s4.fills[i].crop_rgb),
                    width = "stretch",
                    caption = "Container crop"
                )
 
                with c2:
                    if em.ambiguous:
                        st.warning("Food type ambiguous — please confirm.",
                                   icon = "⚠️")
                    if em.low_confidence:
                        st.warning(
                            f"Fill level confidence low "
                            f"({s4.fills[i].confidence:.0%}). "
                            "Consider retaking.",
                            icon = "⚠️"
                        )
                    if em.snap_warning:
                        st.info(
                            "Measured dimensions differed from label — "
                            "label used for volume.",
                            icon = "ℹ️"
                        )
 
                    st.markdown(
                        f"**GN size:** `{em.gn_id}`  "
                        f"({em.container_vol_l:.2f} L total)"
                    )
                    st.markdown(
                        f"**Fill:** `{em.fill_label}` → "
                        f"{em.fill_ratio:.0%} = **{em.food_volume_l:.2f} L**"
                    )
                    st.markdown(
                        f"**Food:** `{em.food_type}`  "
                        f"(density {em.density_kg_l} kg/L)"
                    )
                    st.markdown(f"**Mass:** **{em.mass_kg:.2f} kg**")
                    st.markdown(
                        f"**CO\u2082:** **{em.co2_kg:.3f} kg CO\u2082e**  "
                        f"(factor {em.emission_factor} kg/kg)"
                    )
 
        st.divider()
        st.metric(
            label = f"Total CO\u2082 — {photo_label}",
            value = f"{s9.total_co2_kg:.3f} kg CO\u2082e"
        )


# =============================================================================
# Main layout — Tabs: Photo 1, Photo 2 and comparison
# =============================================================================

st.title("Food Be Good - GN Food Estimator")
st.caption("Estimate food volume and CO\u2082 emissions from GN containers.")

tab1, tab2, tab_cmp = st.tabs([
    "📷  Photo 1 (before)",
    "📷  Photo 2 (after)",
    "📊  CO\u2082 comparison"
    ])


def photo_tab(tab, key: str, label: str):
    with tab:
        st.subheader(f"Capture or upload — {label}")
        method = st.radio(
            "Input method",
            ["Upload file", "Use camera"],
            horizontal = True,
            key = f"method_{key}"
        )
        source = None
        if method == "Upload file":
            up = st.file_uploader(
                "Upload a photo of the GN container",
                type = ["jpg", "jpeg", "png"],
                key = f"upload_{key}"
            )
            if up:
                source = up.read()
        else:
            cam = st.camera_input(
                "Hold camera perpendicular, 20-30 cm above the container.",
                key = f"cam_{key}"
            )
            if cam:
                source = cam.getvalue()
 
        if source is None:
            st.info("No image provided yet.")
            return
 
        # ------------------------------------------------------------------
        # Pass 1: Stages I–VII
        # Run once and cache in session state so re-runs after override
        # do not re-execute the expensive CV stages.
        # ------------------------------------------------------------------
        pass1_key = f"pass1_{key}"
        if st.session_state.get(f"source_hash_{key}") != hash(source):
            with st.spinner("Running pipeline (Stages I-VII)…"):
                partial = run_pipeline_pass1(source)
            if partial is None:
                return
            
            st.session_state[pass1_key] = partial
            st.session_state[f"source_hash_{key}"] = hash(source)

            # Clear any previous overrides and final results when new image arrives
            st.session_state.pop(f"overrides_{key}", None)
            st.session_state.pop(f"results_{key}", None)

        partial = st.session_state.get(pass1_key)
        if partial is None:
            return

        # ------------------------------------------------------------------
        # Override UI: show food ID results and let user correct if needed
        # ------------------------------------------------------------------
        s7 = partial["s7"]
        s4 = partial["s4"]
        overrides = st.session_state.get(f"overrides_{key}", {})
        needs_override = any(fi.ambiguous for fi in s7.food_ids)

        if needs_override or overrides:
            st.divider()
            st.markdown("#### Food identification — please confirm")

        for i, fi in enumerate(s7.food_ids):
            if fi.ambiguous or i in overrides:
                with st.container():
                    oc1, oc2 = st.columns([1, 3])
                    oc1.image(
                        to_pil(s4.fills[i].crop_rgb),
                        width = "stretch",
                        caption = f"Container {i+1}"
                    )
                    with oc2:
                        if fi.ambiguous and i not in overrides:
                            st.warning(
                                f"Container {i+1}: food type could not be "
                                f"identified automatically (top prediction: "
                                f"'{fi.top_k[0]['label'] if fi.top_k else 'none'}', "
                                f"score {fi.food_confidence:.0%}). "
                                "Please select the correct food below.",
                                icon = "⚠️"
                            )
                        elif i in overrides:
                            st.info(
                                f"Container {i+1}: manually set to "
                                f"**{overrides[i]}**.",
                                icon = "✅"
                            )

                        # Top-k predictions for reference
                        if fi.top_k:
                            with st.expander("Model predictions", expanded = False):
                                for p in fi.top_k:
                                    mapped = p.get("mapped") or "not mapped"
                                    st.markdown(
                                        f"- `{p['label']}` "
                                        f"({p['score']:.0%}) → {mapped}"
                                    )

                        current = overrides.get(i, fi.food_type
                                  if fi.food_type != "unknown"
                                  else cfg.SUPPORTED_FOODS[0])
                        choice = st.selectbox(
                            f"Food type for container {i+1}",
                            options = cfg.SUPPORTED_FOODS,
                            index = cfg.SUPPORTED_FOODS.index(current)
                                  if current in cfg.SUPPORTED_FOODS else 0,
                            key = f"override_{key}_{i}"
                        )
                        overrides[i] = choice

        # Always show a manual override option even when auto-detection succeeded
        with st.expander("Manually override food type", expanded = False):
            for i, fi in enumerate(s7.food_ids):
                if not fi.ambiguous and i not in overrides:
                    current = fi.food_type
                    choice = st.selectbox(
                        f"Container {i+1} — detected: `{current}`",
                        options = cfg.SUPPORTED_FOODS,
                        index = cfg.SUPPORTED_FOODS.index(current)
                              if current in cfg.SUPPORTED_FOODS else 0,
                        key = f"manual_{key}_{i}"
                    )
                    if choice != current:
                        overrides[i] = choice

        st.session_state[f"overrides_{key}"] = overrides

        # ------------------------------------------------------------------
        # Pass 2: Stages VIII–IX (runs whenever overrides change)
        # ------------------------------------------------------------------
        with st.spinner("Calculating mass and CO₂…"):
            results = run_pipeline_pass2(partial, overrides)

        st.success(f"Pipeline complete — {label}.")
        show_photo_results(results, label)
        st.session_state[f"results_{key}"] = results
 
 
photo_tab(tab1, "p1", "Photo 1 (before)")
photo_tab(tab2, "p2", "Photo 2 (after)")


# =============================================================================
# Stage XI — CO₂ saved comparison
# =============================================================================

with tab_cmp:
    st.subheader("CO\u2082 savings — Stage XI")
 
    r1 = st.session_state.get("results_p1")
    r2 = st.session_state.get("results_p2")
 
    if r1 is None or r2 is None:
        st.info("Process both photos to see the CO\u2082 savings comparison.")
    else:
        comp = stage10_11_compare.run(r1["s9"], r2["s9"])
 
        # Summary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("CO\u2082 before", f"{comp.total_co2_before:.3f} kg CO\u2082e")
        m2.metric("CO\u2082 after", f"{comp.total_co2_after:.3f} kg CO\u2082e")
 
        saved = comp.total_co2_saved
        if saved >= 0:
            m3.metric(
                "CO\u2082 saved",
                f"{saved:.3f} kg CO\u2082e",
                delta = f"-{saved:.3f} kg",
                delta_color = "inverse"
            )
        else:
            m3.metric(
                "CO\u2082 increase",
                f"{abs(saved):.3f} kg CO\u2082e",
                delta = f"+{abs(saved):.3f} kg",
                delta_color = "normal"
            )
 
        # Per-container cards
        st.divider()
        st.markdown("#### Per-container breakdown")
 
        for i, c in enumerate(comp.containers):
            with st.expander(
                f"Container {i+1} — {c.container_label} ({c.gn_id})",
                expanded = True
            ):
                ca, cb, cc = st.columns(3, border = True)
                with ca:
                    st.markdown("**Before (Photo 1)**")
                    st.markdown(f"Food: `{c.food_type_before}`")
                    st.markdown(f"Fill: `{c.fill_before}`")
                    st.markdown(f"Mass: {c.mass_kg_before:.2f} kg")
                    st.markdown(f"CO\u2082: {c.co2_kg_before:.3f} kg")
                
                with cb:
                    st.markdown("**After (Photo 2)**")
                    st.markdown(f"Food: `{c.food_type_after}`")
                    st.markdown(f"Fill: `{c.fill_after}`")
                    st.markdown(f"Mass: {c.mass_kg_after:.2f} kg")
                    st.markdown(f"CO\u2082: {c.co2_kg_after:.3f} kg")
                
                with cc:
                    st.markdown("**Difference**")
                    mass_diff = c.mass_kg_before - c.mass_kg_after
                    st.markdown(f"Mass used: {mass_diff:.2f} kg")
                    if c.co2_saved_kg >= 0:
                        st.markdown(
                            f"CO\u2082 saved: "
                            f":green[**{c.co2_saved_kg:.3f} kg**]"
                        )
                    else:
                        st.markdown(
                            f"CO\u2082 increase: "
                            f":red[**{abs(c.co2_saved_kg):.3f} kg**]"
                        )
 
        # Summary table
        st.divider()
        st.markdown("#### Summary table")
 
        rows = []
        for c in comp.containers:
            rows.append({
                "Container":        c.container_label,
                "GN ID":            c.gn_id,
                "Food before":      c.food_type_before,
                "Food after":       c.food_type_after,
                "Fill before":      c.fill_before,
                "Fill after":       c.fill_after,
                "Mass before (kg)": round(c.mass_kg_before, 3),
                "Mass after (kg)":  round(c.mass_kg_after,  3),
                "CO\u2082 before (kg)": round(c.co2_kg_before, 3),
                "CO\u2082 after (kg)":  round(c.co2_kg_after,  3),
                "CO\u2082 saved (kg)":  round(c.co2_saved_kg,  3)
            })
        rows.append({
            "Container": "Total",
            "GN ID": "",
            "Food before": "",
            "Food after": "",
            "Fill before": "",
            "Fill after": "",
            "Mass before (kg)": round(comp.total_mass_before, 3),
            "Mass after (kg)":  round(comp.total_mass_after,  3),
            "CO\u2082 before (kg)": round(comp.total_co2_before, 3),
            "CO\u2082 after (kg)":  round(comp.total_co2_after,  3),
            "CO\u2082 saved (kg)":  round(comp.total_co2_saved,  3)
        })
 
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width = True,
            hide_index = True
        )
        st.caption(cfg.CO2_DISCLAIMER)
        