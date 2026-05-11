import streamlit as st
import requests
import json
import joblib
import pandas as pd
from pathlib import Path

# Paths
GUI_DIR = Path(__file__).parent.resolve()
ROOT_DIR = GUI_DIR.parent
MODELS_DIR = ROOT_DIR / "models" / "saved"

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Clinical Intelligence", layout="wide")

st.title("🏥 Clinical Intelligence Dashboard")
st.markdown("Modern AI-based clinical decision support system featuring explainable AI (SHAP), clinical rule integration, and data drift monitoring.")

# Navigation
page = st.sidebar.radio("Navigation", ["Patient Risk Prediction", "XAI Comparison", "Experiment Tracking"])

def get_default_features(disease):
    if disease == "stroke":
        return {
            "gender": 1,
            "age": 67.0,
            "hypertension": 1,
            "heart_disease": 1,
            "ever_married": 1,
            "Residence_type": 1,
            "avg_glucose_level": 228.69,
            "bmi": 36.6,
            "work_type_Govt_job": 0,
            "work_type_Never_Worked": 0,
            "work_type_Private": 1,
            "work_type_Self-Employed": 0,
            "smoking_status_formerly smoked": 1,
            "smoking_status_never smoked": 0,
            "smoking_status_smokes": 0
        }
    elif disease == "diabetes":
        return {
            "Age": 50,
            "Gender": 1,
            "Polyuria": 1,
            "Polydipsia": 1,
            "sudden weight loss": 0,
            "weakness": 1,
            "Polyphagia": 0,
            "Genital thrush": 0,
            "visual blurring": 0,
            "Itching": 0,
            "Irritability": 0,
            "delayed healing": 0,
            "partial paresis": 0,
            "muscle stiffness": 0,
            "Alopecia": 0,
            "Obesity": 1
        }
    elif disease == "autism":
        return {
            "A1": 1, "A2": 1, "A3": 1, "A4": 1, "A5": 1,
            "A6": 0, "A7": 1, "A8": 0, "A9": 0, "A10": 1,
            "Age": 26, "Sex": 1, "Jaundice": 0, "Family_ASD": 0,
            "AQ_Score": 7
        }

if page == "Patient Risk Prediction":
    st.header("Patient Risk Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Patient Profile")
        disease = st.selectbox("Select Disease Model", ["autism", "diabetes", "stroke"])
        
        default_data = get_default_features(disease)
        
        features_json = st.text_area("Patient Features (JSON format)", json.dumps(default_data, indent=4), height=350)
        
        if st.button("Predict Risk"):
            try:
                features = json.loads(features_json)
                
                with st.spinner("Analyzing patient profile..."):
                    # Call FastAPI
                    response = requests.post(f"{API_URL}/predict", json={"disease": disease, "features": features})
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state["prediction"] = data
                        st.session_state["current_disease"] = disease
                        st.session_state["last_features"] = features
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                        
            except json.JSONDecodeError:
                st.error("Invalid JSON format in features. Please check your syntax.")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the FastAPI backend. Is it running on port 8000?")
                
    with col2:
        st.subheader("2. AI Insights")
        
        if "prediction" in st.session_state:
            pred_data = st.session_state["prediction"]
            curr_disease = st.session_state["current_disease"]
            
            # Risk Level Card
            risk_color = "red" if pred_data["risk_level"] == "High" else "green"
            st.markdown(f"### Risk Level: <span style='color:{risk_color}'>{pred_data['risk_level']}</span>", unsafe_allow_html=True)
            if pred_data.get("probability") is not None:
                st.progress(pred_data["probability"])
                st.caption(f"Confidence: {pred_data['probability']:.1%}")
                
            st.divider()
            
            st.markdown("#### Explainable AI (SHAP)")
            # Load artifacts for XAI
            explain_path = MODELS_DIR / f"{curr_disease}_explainability.pkl"
            if explain_path.exists():
                xai_data = joblib.load(explain_path)
                
                shap_imp = xai_data.get("shap_importance")
                if shap_imp:
                    # Convert to dataframe for charting
                    df_shap = pd.DataFrame(list(shap_imp.items()), columns=["Feature", "Importance"]).head(10)
                    df_shap.set_index("Feature", inplace=True)
                    st.bar_chart(df_shap)
                else:
                    st.info("SHAP values not found. Showing Tree Importance fallback.")
                    tree_imp = xai_data.get("tree_importance", {})
                    df_tree = pd.DataFrame(list(tree_imp.items()), columns=["Feature", "Importance"]).head(10)
                    df_tree.set_index("Feature", inplace=True)
                    st.bar_chart(df_tree)
                    
                st.divider()
                st.markdown("#### Clinical Rules (Reasoning Layer)")
                rules = xai_data.get("rules", [])
                
                # We just display the relevant registered rules
                for r_id, cond, conc, conf, ev in rules[:3]: # limit to top 3 for brevity
                    with st.expander(f"Rule {r_id}"):
                        st.write(f"**Condition:** {cond}")
                        st.write(f"**Conclusion:** {conc}")
                        
            else:
                st.warning("Explainability artifacts not found. Run models/explainability.py")
        else:
            st.info("Submit a patient profile to see AI insights.")

elif page == "XAI Comparison":
    st.header("XAI Comparison — LIME vs SHAP")
    st.markdown(
        "Compare two complementary Explainable AI techniques. "
        "**SHAP** provides global model-level feature importance averaged across all patients. "
        "**LIME** provides a local explanation for one specific patient."
    )

    disease = st.selectbox("Select Disease Model", ["autism", "diabetes", "stroke"], key="xai_disease")
    explain_path = MODELS_DIR / f"{disease}_explainability.pkl"
    artifact_path = MODELS_DIR / f"{disease}_artifacts.pkl"
    model_path = MODELS_DIR / f"{disease}_best_model.pkl"

    if not explain_path.exists() or not artifact_path.exists():
        st.warning("Artifacts not found. Run `python models/explainability.py` first.")
    else:
        xai_data = joblib.load(explain_path)
        art = joblib.load(artifact_path)
        X_train = art["X_train"]
        X_test = art["X_test"]
        feature_names = art.get("feature_names", [f"f{i}" for i in range(X_train.shape[1])])

        # --- Patient Selection for LIME ---
        st.divider()
        
        # Check if we have a prediction from the other tab
        has_session_patient = (
            "prediction" in st.session_state and
            st.session_state.get("current_disease") == disease
        )
        
        if has_session_patient:
            st.success("🔗 Using the patient you just predicted for in the Prediction tab. Switch patients there and come back to update LIME!")
            lime_source = "session"
        else:
            st.info("💡 Tip: Go to **Patient Risk Prediction**, run a prediction, then come back here to see LIME explain that exact patient.")
            lime_source = "test_index"
        
        patient_idx = st.slider(
            "Or select a test patient by index",
            min_value=0, max_value=len(X_test) - 1, value=0,
            disabled=has_session_patient
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔵 SHAP — Global Feature Importance")
            st.caption("Average contribution across all patients in the test set.")
            shap_imp = xai_data.get("shap_importance", {})
            if shap_imp:
                df_shap = pd.DataFrame(list(shap_imp.items()), columns=["Feature", "SHAP Importance"]).head(10)
                df_shap.set_index("Feature", inplace=True)
                st.bar_chart(df_shap)
            else:
                st.info("SHAP values not available. Re-run explainability.py.")

        with col2:
            st.subheader("🟠 LIME — Local Patient Explanation")
            st.caption("Explains the prediction for this specific patient using a local surrogate model.")
            
            try:
                from lime.lime_tabular import LimeTabularExplainer

                if lime_source == "session":
                    # Build input from the JSON the user entered
                    session_features = st.session_state.get("last_features", {})
                    if session_features:
                        ordered = [session_features.get(f, 0) for f in feature_names]
                        X_sample = __import__("numpy").array(ordered, dtype=float)
                    else:
                        X_sample = X_test[0]
                else:
                    X_sample = X_test[patient_idx]

                model = joblib.load(model_path)
                explainer = LimeTabularExplainer(
                    X_train,
                    feature_names=feature_names,
                    class_names=["Low Risk", "High Risk"],
                    mode="classification",
                    discretize_continuous=False,
                    random_state=42,
                )
                exp = explainer.explain_instance(
                    X_sample,
                    model.predict_proba,
                    num_features=10,
                )
                lime_weights = dict(exp.as_list())
                df_lime = pd.DataFrame(
                    sorted(lime_weights.items(), key=lambda x: abs(x[1]), reverse=True),
                    columns=["Feature", "LIME Weight"]
                ).head(10)
                df_lime.set_index("Feature", inplace=True)
                st.bar_chart(df_lime)
                st.caption("Positive = pushes toward High Risk | Negative = pushes toward Low Risk")
            except ImportError:
                st.error("LIME not installed. Run `pip install lime`.")
            except Exception as e:
                st.error(f"LIME computation failed: {e}")

        st.divider()
        st.markdown("#### Key Difference")
        st.markdown(
            "| | SHAP | LIME |\n"
            "|---|---|---|\n"
            "| **Scope** | Global (all patients) | Local (one patient) |\n"
            "| **Method** | Shapley game theory | Local surrogate model |\n"
            "| **Output** | Feature importance scores | Feature weight rules |\n"
            "| **Best for** | Model auditing & trust | Patient-level explanation |"
        )

elif page == "Experiment Tracking":
    st.header("MLOps Experiment Tracking")
    st.markdown("Track and compare model performance metrics across different datasets and architectures.")

    results_path = MODELS_DIR / "ensemble_results.csv"
    if not results_path.exists():
        st.warning("No experiment logs found. Run `python models/ensemble_models.py` first to generate metrics.")
    else:
        df_results = pd.read_csv(results_path)
        
        # Format metrics for display
        display_cols = ["Model", "CV F1", "Accuracy", "F1", "Recall", "Precision", "ROC-AUC"]
        
        datasets = df_results["Dataset"].unique() if "Dataset" in df_results.columns else ["autism", "diabetes", "stroke"]
        selected_ds = st.selectbox("Filter by Dataset", ["All"] + list(datasets))
        
        if selected_ds != "All":
            df_display = df_results[df_results["Dataset"] == selected_ds]
        else:
            df_display = df_results
            
        st.dataframe(df_display, use_container_width=True)
        
        st.divider()
        st.subheader("Performance Comparison")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**F1-Score Comparison** (Higher is better)")
            # Pivot table for plotting
            if "Dataset" in df_display.columns:
                pivot_f1 = df_display.pivot(index="Model", columns="Dataset", values="F1")
                st.bar_chart(pivot_f1)
            else:
                st.bar_chart(df_display.set_index("Model")["F1"])
                
        with col2:
            st.markdown("**ROC-AUC Comparison** (Higher is better)")
            if "Dataset" in df_display.columns:
                pivot_roc = df_display.pivot(index="Model", columns="Dataset", values="ROC-AUC")
                st.bar_chart(pivot_roc)
            else:
                st.bar_chart(df_display.set_index("Model")["ROC-AUC"])
