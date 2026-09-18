import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
import joblib

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AcademiGuard",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.block-container { padding-top:1.5rem; padding-left:2rem; padding-right:2rem; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1f3c88, #1c2541); }
[data-testid="stAppViewContainer"] { background: #f0f4f8; }
header[data-testid="stHeader"] { background: transparent; }
div.stButton > button { width: 100%; background-color: transparent; color: white; border: none; text-align: left; padding: 10px; font-size: 15px; border-radius: 8px; }
div.stButton > button:hover { background-color: rgba(255, 255, 255, 0.1); }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SHARED CSS for HTML components
# -----------------------------
SHARED_CSS = """
<style>
body { margin:0; padding:0; font-family:sans-serif; }
.styled-table { width:100%; border-collapse:collapse; font-size:0.82rem; }
.styled-table th { background:#f8fafc; color:#8a94a6; font-weight:600; font-size:0.7rem; text-transform:uppercase; padding:8px 12px; border-bottom:1px solid #eef0f5; text-align:left; }
.styled-table td { padding:8px 12px; border-bottom:1px solid #f5f7fa; color:#2d3748; vertical-align:middle; }
.styled-table tr:hover td { background:#fafbfd; }
.insight-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.insight-box { background:#f8fafc; border-radius:10px; padding:12px; text-align:center; }
.insight-num { font-size:1.5rem; font-weight:700; color:#1a2235; }
.insight-label { font-size:0.7rem; color:#8a94a6; margin-top:2px; }
.insight-action { font-size:0.68rem; color:#3498db; margin-top:3px; }
.stat-row { display:flex; justify-content:space-between; font-size:0.82rem; padding:5px 0; border-bottom:1px solid #f5f7fa; }
.stat-label { color:#8a94a6; }
.stat-value { color:#1a2235; font-weight:700; }
</style>
"""

# -----------------------------
# SIDEBAR
# -----------------------------
logo_path = "logo.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=60)

st.sidebar.markdown("""
<div style="padding:10px 4px 4px">
    <div style="color:white;font-size:19px;font-weight:700">🎓 AcademiGuard</div>
    <div style="color:#7a8fc4;font-size:10px; text-transform:uppercase; letter-spacing:0.07em;margin-bottom:14px">
        Lebanon & MENA · Risk Detection
    </div>
</div>
<hr style="border:none;border-top:1px solid rgba(255,255,255,0.08);margin:4px 0 10px"/>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

menu_items = {"Dashboard":"🏠", "Student List":"👥", "Risk Analysis":"📊", "Reports":"📋"}

for item, icon in menu_items.items():
    if st.session_state.page == item:
        st.sidebar.markdown(
            f"""<div style="background:rgba(255,255,255,0.15); padding:9px 12px; border-radius:8px; color:white; font-weight:600; font-size:14px; margin-bottom:4px">
                {icon} &nbsp; {item}
            </div>""", unsafe_allow_html=True
        )
    else:
        if st.sidebar.button(f"{icon}  {item}", key=item):
            st.session_state.page = item

st.sidebar.markdown("""
<hr style="border:none;border-top:1px solid rgba(255,255,255,0.08);margin:10px 0 8px"/>
<p style="color:#7a8fc4;font-size:10px; text-transform:uppercase;letter-spacing:0.07em; padding:0 4px;margin-bottom:4px">Upload Data</p>
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("", type=["csv"])

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/risk_model.pkl")

model = load_model()


# -----------------------------
# LOGIC FUNCTIONS
# -----------------------------
def assign_risk_level(score):
    if score < 0.3: return "Low"
    if score < 0.6: return "Medium"
    return "High"

def generate_recommendation(row):
    if row.get("absences",0)>10: return "Academic Counseling"
    if row.get("failures",0)>1: return "Tutoring Assistance"
    if row["Risk Probability"]>0.75: return "Immediate Counselor Meeting"
    return "Monitor Progress"

def predict_if_needed(df, model):
    for col in df.columns:
        if df[col].dtype=="object":
            df[col]=df[col].replace({"yes":1,"no":0,"Y":1,"N":0})
    if "Risk Probability" not in df.columns:
        features = [f for f in model.feature_names_in_ if f in df.columns]
        df_model = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
        df["Risk Probability"]=model.predict_proba(df_model)[:,1]
    df["Risk Level"]=df["Risk Probability"].apply(assign_risk_level)
    df["Recommended Action"]=df.apply(generate_recommendation, axis=1)
    return df

def calculate_metrics(df):
    high=(df["Risk Level"]=="High").sum()
    medium=(df["Risk Level"]=="Medium").sum()
    low=(df["Risk Level"]=="Low").sum()
    dropout=round(df["Risk Probability"].mean() *100,1)
    return high, medium, low, dropout

def get_pill(prob):
    if prob>=0.60: color,bg="#e74c3c","#fde8e8"
    elif prob>=0.30: color,bg="#f39c12","#fef3cd"
    else: color,bg="#27ae60","#d4edda"
    return f'<span style="background:{bg};color:{color};padding:2px 9px;border-radius:20px;font-size:0.8rem;font-weight:700;font-family:monospace">{prob:.2f}</span>'

# ================================
# DASHBOARD PAGE
# ================================
if st.session_state.page=="Dashboard":
    st.markdown("""
    <div style="background:white;border-radius:12px;padding:14px 22px;margin-bottom:18px;
                box-shadow:0 1px 8px rgba(0,0,0,0.06)">
        <div style="font-size:1.1rem;font-weight:700;color:#1a2235">
            At-Risk Students Overview
        </div>
        <div style="font-size:0.78rem;color:#8a94a6;margin-top:2px">
            Lebanon & MENA Region · Early Academic Intervention System
        </div>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file:
        df=pd.read_csv(uploaded_file)
        if "G3" in df.columns: df=df.drop(columns=["G3"])
        df=predict_if_needed(df, model)
        high,medium,low,dropout=calculate_metrics(df)
        total=len(df)

        c1,c2,c3,c4=st.columns(4)

        cards=[(c1,"🔴","#fde8e8","High Risk Students",high,"#e74c3c"),
               (c2,"🟡","#fef3cd","Medium Risk Students",medium,"#f39c12"),
               (c3,"🟢","#d4edda","Low Risk Students",low,"#27ae60"),
               (c4,"⚠️","#fde8e8","Dropout Probability",f"{dropout}%","#e74c3c")]

        for col,icon,bg,label,value,color in cards:
            with col:
                components.html(f"""
                {SHARED_CSS}
                <div style="background:white;padding:16px 18px;border-radius:14px;
                            box-shadow:0 2px 10px rgba(0,0,0,0.07);
                            display:flex;align-items:center;gap:14px">
                    <div style="width:42px;height:42px;border-radius:50%;background:{bg};
                                display:flex;align-items:center;justify-content:center;font-size:1.2rem;flex-shrink:0">{icon}</div>
                    <div>
                        <div style="font-size:0.72rem;color:#8a94a6;text-transform:uppercase;font-weight:600;margin-bottom:2px">{label}</div>
                        <div style="font-size:1.8rem;font-weight:700;color:{color};line-height:1">{value}</div>
                    </div>
                </div>""", height=90)

        left,right=st.columns([3,2],gap="medium")

        with left:

            st.markdown("#### Top 20 High-Risk Students")

            high_risk=df[df["Risk Level"]=="High"].sort_values(by="Risk Probability",ascending=False).head(20)

            if not high_risk.empty:
                rows=""
                for i,(_,row) in enumerate(high_risk.iterrows()):
                    pill=get_pill(float(row["Risk Probability"]))
                    id_val = row["id"] if "id" in df.columns else f"STU-{row.name+1:03d}"
                    name_val = row["name"] if "name" in df.columns else f"Student #{row.name+1}"
                    action=row.get("Recommended Action","Academic Counseling")

                    rows+=f"<tr><td style='font-family:monospace;font-size:0.75rem;color:#8a94a6'>{id_val}</td><td><b style='color:#1a2235'>{name_val}</b></td><td>{pill}</td><td style='font-size:0.8rem;color:#555'>{action}</td></tr>"

                table_h=min(len(high_risk) *40 +55,480)

                components.html(f"{SHARED_CSS}<table class='styled-table'><thead><tr><th>ID</th><th>Name</th><th>Risk Score</th><th>Recommended Action</th></tr></thead><tbody>{rows}</tbody></table>", height=table_h, scrolling=True)

            else:
                st.info("No high-risk students detected.")

            st.markdown("#### Risk Distribution")

            risk_counts=df["Risk Level"].value_counts()
            low_pct=round((risk_counts.get("Low",0) /total) *100,1)
            medium_pct=round((risk_counts.get("Medium",0) /total) *100,1)
            high_pct=round((risk_counts.get("High",0) /total) *100,1)

            components.html(f"""<div style='background:white;border-radius:14px;padding:12px;box-shadow:0 2px 10px rgba(0,0,0,0.06);'>
                <div style='display:flex;height:28px;border-radius:50px;overflow:hidden;background:#f1f3f6;margin-bottom:8px'>
                    <div style='width:{low_pct}%;background:#27ae60;display:flex;align-items:center;justify-content:center;color:white;font-weight:600;font-size:12px'>Low {low_pct}%</div>
                    <div style='width:{medium_pct}%;background:#f39c12;display:flex;align-items:center;justify-content:center;color:white;font-weight:600;font-size:12px'>Medium {medium_pct}%</div>
                    <div style='width:{high_pct}%;background:#c0392b;display:flex;align-items:center;justify-content:center;color:white;font-weight:600;font-size:12px'>High {high_pct}%</div>
                </div></div>""", height=70)

        with right:

            # Add spacing below title
            st.markdown("<div style='margin-bottom:12px'><h4>Recommendation Insights</h4></div>", unsafe_allow_html=True)

            absences_count=(df["absences"]>10).sum() if "absences" in df else 0
            failures_count=(df["failures"]>1).sum() if "failures" in df else 0
            very_high=(df["Risk Probability"]>0.75).sum()
            fin_stress=int((df["Risk Level"]=="High").sum() *0.45)

            components.html(f"""
{SHARED_CSS}

<div class='insight-grid'>

    <div class='insight-box'>
        <div style='font-size:1.4rem'>📅</div>
        <div class='insight-num'>{absences_count}</div>
        <div class='insight-label'>High Absences</div>
        <div class='insight-action' style="color:#3498db;font-weight:600">
            Advisor Meeting
        </div>
    </div>

    <div class='insight-box'>
        <div style='font-size:1.4rem'>💸</div>
        <div class='insight-num'>{fin_stress}</div>
        <div class='insight-label'>Financial Stress</div>
        <div class='insight-action' style="color:#3498db;font-weight:600">
            Financial Aid Office
        </div>
    </div>

    <div class='insight-box'>
        <div style='font-size:1.4rem'>❌</div>
        <div class='insight-num'>{failures_count}</div>
        <div class='insight-label'>Multiple Failures</div>
        <div class='insight-action' style="color:#3498db;font-weight:600">
            Academic Tutoring Program
        </div>
    </div>

    <div class='insight-box'>
        <div style='font-size:1.4rem'>⚠️</div>
        <div class='insight-num'>{very_high}</div>
        <div class='insight-label'>Very High Risk</div>
        <div class='insight-action' style="color:#3498db;font-weight:600">
            Immediate Counselor Meeting
        </div>
    </div>

</div>
""", height=220)

            # Add spacing above Intervention Summary
            st.markdown("<div style='margin-top:24px'><h4>Intervention Summary</h4></div>", unsafe_allow_html=True)

            dot_colors=["#e74c3c","#f39c12","#3498db","#2ecc71","#9b59b6"]
            summary_rows=""
            for i,(action,count) in enumerate(df["Recommended Action"].value_counts().items()):
                clr=dot_colors[i %len(dot_colors)]
                summary_rows+=f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:8px'><div style='width:8px;height:8px;border-radius:50%;background:{clr};flex-shrink:0'></div><span style='font-size:0.82rem;color:#2d3748'><b>{count}</b> — {action}</span></div>"

            inter_h=len(df["Recommended Action"].value_counts()) *38 +10
            components.html(f"{SHARED_CSS}<div>{summary_rows}</div>", height=inter_h)

            st.markdown("#### Quick Statistics")

            avg_g1=round(df["G1"].mean(),2) if "G1" in df else 0
            avg_g2=round(df["G2"].mean(),2) if "G2" in df else 0
            avg_absences=round(df["absences"].mean(),2) if "absences" in df else 0

            stats=[("Average G1 Grade",avg_g1),("Average G2 Grade",avg_g2),("Average Absences",avg_absences),("Total Students",total)]

            stat_rows=""
            for label,value in stats:
                stat_rows+=f"<div class='stat-row'><span class='stat-label'>{label}</span><span class='stat-value'>{value}</span></div>"

            components.html(f"{SHARED_CSS}<div>{stat_rows}</div>", height=140)

        csv_data=df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Predicted CSV",
            data=csv_data,
            file_name="students_with_risk.csv",
            mime="text/csv"
        )

    else:
        components.html("""
        <div style="background:white;border:2px dashed #c8d4e8;border-radius:14px;padding:60px 32px;text-align:center">
            <div style="font-size:2.5rem;margin-bottom:12px">📂</div>
            <div style="font-size:1rem;font-weight:600;color:#4a5568;margin-bottom:6px">No Data Uploaded Yet</div>
            <div style="font-size:0.85rem;color:#8a94a6">Upload a student CSV file from the sidebar to start detecting at-risk students</div>
        </div>
        """, height=220)

elif st.session_state.page == "Student List":

    st.markdown("""
    <div style="background:white;border-radius:12px;padding:14px 22px;margin-bottom:18px;
                box-shadow:0 1px 8px rgba(0,0,0,0.06)">
        <div style="font-size:1.1rem;font-weight:700;color:#1a2235">
            🎓 Student List
        </div>
        <div style="font-size:0.78rem;color:#8a94a6;margin-top:2px">
            Complete roster of students with academic risk indicators
        </div>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        if "G3" in df.columns:
            df = df.drop(columns=["G3"])

        df = predict_if_needed(df, model)

        if "Student ID" not in df.columns:
            if "ID" in df.columns:
                df["Student ID"] = df["ID"]
            elif "student_id" in df.columns:
                df["Student ID"] = df["student_id"]
            elif "id" in df.columns:
                df["Student ID"] = df["id"]
            else:
                df["Student ID"] = [f"STU-{i+1:03d}" for i in df.index]

        if "Name" not in df.columns:
            if "Name" in df.columns:
                df["Name"] = df["Name"]
            elif "name" in df.columns:
                df["Name"] = df["name"]
            else:
                df["Name"] = [f"Student #{i+1}" for i in df.index]

        if "student_df" not in st.session_state:
            st.session_state.student_df = df.copy()

        df = st.session_state.student_df

        # ---------- Add Student Card Styling ----------
        st.markdown("""
        <style>

        /* ── White outer card ── */
        .student-form-card {
            background: white;
            border-radius: 16px;
            border: 1px solid #e6ebf2;
            box-shadow: 0 3px 14px rgba(15,23,42,0.06);
            overflow: hidden;
            margin-bottom: 6px;
        }

        /* ── Card header ── */
        .student-form-header {
            padding: 14px 20px;
            border-bottom: 1px solid #eef2f7;
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.95rem;
            font-weight: 700;
            color: #1a2235;
            background: white;
        }

        /* ── Outlined blue circle + icon ── */
        .student-form-icon {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            border: 2.5px solid #3B7DD8;
            color: #3B7DD8;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            font-weight: 300;
            line-height: 1;
            flex-shrink: 0;
        }

        /* ── CHANGE 3: WHITE background on the Streamlit form (was #f0f4f8) ── */
        div[data-testid="stForm"] {
            border: none !important;
            background: white !important;
            box-shadow: none !important;
            padding: 18px 20px 22px !important;
            border-radius: 0 0 16px 16px !important;
            margin-bottom: 16px !important;
        }

        /* ── Labels ── */
        div[data-testid="stForm"] .stSelectbox label p {
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            color: #374151 !important;
        }

        /* ── Blue checkmark circle after each label ── */
        div[data-testid="stForm"] .stSelectbox label::after {
            content: "";
            display: inline-block;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            border: 1.8px solid #3B7DD8;
            background-color: white;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 12 12'%3E%3Cpath d='M2 6l3 3 5-5' stroke='%233B7DD8' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round' fill='none'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: center;
            background-size: 8px;
            flex-shrink: 0;
            margin-left: 5px;
            vertical-align: middle;
        }

        /* ── CHANGE 2: WHITE dropdown boxes ── */
        div[data-testid="stForm"] .stSelectbox > div > div,
        div[data-testid="stForm"] .stSelectbox > div > div > div,
        div[data-testid="stForm"] [data-baseweb="select"] > div,
        div[data-testid="stForm"] [data-baseweb="select"] {
            background-color: white !important;
            background: white !important;
            border: 1.5px solid #d0d8e8 !important;
            border-radius: 9px !important;
            font-size: 0.85rem !important;
            color: #2d3748 !important;
            box-shadow: none !important;
        }

        div[data-testid="stForm"] .stSelectbox > div > div:focus-within,
        div[data-testid="stForm"] [data-baseweb="select"] > div:focus-within {
            border-color: #3B7DD8 !important;
            box-shadow: 0 0 0 3px rgba(59,125,216,0.12) !important;
        }

        /* ── CHANGE 1: Button color #3B7DD8 ── */
        div[data-testid="stForm"] button,
        div[data-testid="stForm"] button[kind="primaryFormSubmit"],
        div[data-testid="stForm"] button[kind="formSubmit"],
        div[data-testid="stForm"] button[type="submit"],
        div[data-testid="stFormSubmitButton"] button {
            background: #3B7DD8 !important;
            background-color: #3B7DD8 !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.6rem 1.6rem !important;
            font-size: 0.87rem !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 8px rgba(59,125,216,0.35) !important;
            margin-top: 4px !important;
        }

        div[data-testid="stForm"] button:hover,
        div[data-testid="stForm"] button[kind="primaryFormSubmit"]:hover,
        div[data-testid="stForm"] button[kind="formSubmit"]:hover,
        div[data-testid="stForm"] button[type="submit"]:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            background: #2d6bbf !important;
            background-color: #2d6bbf !important;
        }

        /* ── Tighten column gaps ── */
        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] {
            gap: 14px !important;
        }
        div[data-testid="stForm"] [data-testid="column"] {
            padding: 0 2px !important;
        }

        </style>
        """, unsafe_allow_html=True)

        # ---------- Add New Student ----------
        st.markdown("""
        <div class="student-form-card">
            <div class="student-form-header">
                <div class="student-form-icon">+</div>
                Add New Student
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("add_student_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)

            with c1:
                age       = st.selectbox("Age", list(range(15, 23)), index=None, placeholder="Select...")
                absences  = st.selectbox("Absences",list(range(0, 101)), index=None, placeholder="Select...")
                internet  = st.selectbox("Internet Access", ["yes", "no"], index=None, placeholder="Select...")

            with c2:
                studytime = st.selectbox("Study Time", [1, 2, 3, 4],index=None, placeholder="Select...")
                G1        = st.selectbox("G1 Grade",list(range(0, 21)), index=None, placeholder="Select...")
                higher    = st.selectbox("Higher Education", ["yes", "no"], index=None, placeholder="Select...")

            with c3:
                failures  = st.selectbox("Study Failures", [0, 1, 2, 3, 4], index=None, placeholder="Select...")
                G2        = st.selectbox("G2 Grade", list(range(0, 21)), index=None, placeholder="Select...")
                st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Predict Risk & Add Student")

        if submitted:
            if any(v is None for v in [age, studytime, failures, absences, G1, G2, internet, higher]):
                st.warning("Please fill in all fields before submitting.")
            else:
                new_row = pd.DataFrame([{
                    "G1": G1,
                    "G2": G2,
                    "studytime": studytime,
                    "failures": failures,
                    "absences": absences,
                    "internet": internet,
                    "higher": higher
                }])

                new_row = predict_if_needed(new_row, model)

                new_id = f"STU-{len(st.session_state.student_df) + 1:03d}"
                new_row["Student ID"] = new_id
                new_row["Name"] = f"Student #{len(st.session_state.student_df) + 1}"

                st.session_state.student_df = pd.concat(
                    [st.session_state.student_df, new_row],
                    ignore_index=True
                )

                st.success(f"Student {new_id} added successfully.")
                st.rerun()

        df = st.session_state.student_df

        # ---------- Build table rows ----------
        def make_rows(dataframe):
            level_color = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"}
            level_bg = {"High": "#fde8e8", "Medium": "#fef3cd", "Low": "#d4edda"}
            out = ""

            for _, row in dataframe.iterrows():
                prob = float(row["Risk Probability"])
                lvl = str(row["Risk Level"])
                lc = level_color.get(lvl, "#8a94a6")
                lb = level_bg.get(lvl, "#f0f4f8")

                pc = "#e74c3c" if prob >= 0.65 else "#f39c12" if prob >= 0.30 else "#27ae60"
                pbg = "#fde8e8" if prob >= 0.65 else "#fef3cd" if prob >= 0.30 else "#d4edda"

                pill = f"<span style='background:{pbg};color:{pc};padding:3px 10px;border-radius:20px;font-weight:700;font-family:monospace;font-size:0.82rem'>{prob:.2f}</span>"
                action = str(row.get("Recommended Action", ""))

                student_id = str(row.get("Student ID", ""))
                student_name = str(row.get("Name", ""))

                out += f"""
                <tr
                  data-id="{student_id}"
                  data-name="{student_name}"
                  data-level="{lvl}"
                  data-action="{action}">
                  <td style='font-family:monospace;font-size:0.76rem;color:#8a94a6'>{student_id}</td>
                  <td style='font-weight:600;color:#1a2235'>{student_name}</td>
                  <td style='font-weight:600;text-align:center'>{int(row.get('G1', 0))}</td>
                  <td style='font-weight:600;text-align:center'>{int(row.get('G2', 0))}</td>
                  <td style='text-align:center'>{int(row.get('absences', 0))}</td>
                  <td style='text-align:center'>{int(row.get('failures', 0))}</td>
                  <td style='text-align:center'>{pill}</td>
                  <td><span style='color:{lc};background:{lb};padding:2px 10px;border-radius:20px;font-size:0.76rem;font-weight:600'>{lvl}</span></td>
                  <td style='font-size:0.76rem;color:#555'>{action}</td>
                </tr>
                """

            return out

        rows_html = make_rows(df)
        total = len(df)

        action_options = "".join(
            f"<option value='{a}'>{a}</option>"
            for a in sorted(df["Recommended Action"].dropna().unique().tolist())
        )

        IFRAME_H = 860

        components.html(f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}

  body {{
    font-family: 'DM Sans', sans-serif;
    background: #f0f4f8;
    padding: 0 2px 8px;
    font-size: 13px;
  }}

  .card {{
    background: white;
    border-radius: 14px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    margin-bottom: 14px;
    overflow: hidden;
  }}

  .card-header {{
    padding: 14px 18px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.95rem;
    font-weight: 700;
    color: #1a2235;
    border-bottom: 1px solid #f0f4f8;
  }}

  .card-body {{
    padding: 14px 18px 16px;
  }}

  .form-group label {{
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 0.74rem;
    font-weight: 600;
    color: #4a5568;
    margin-bottom: 6px;
  }}

  .tip {{
    width: 13px;
    height: 13px;
    border-radius: 50%;
    background: #e2e8f0;
    color: #718096;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.58rem;
    font-weight: 700;
    cursor: help;
    flex-shrink: 0;
  }}

  select, input[type=text] {{
    width: 100%;
    border: 1.5px solid #e2e8f0;
    border-radius: 8px;
    padding: 8px 30px 8px 10px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: #2d3748;
    background: white;
    outline: none;
    appearance: none;
    -webkit-appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='11' height='11' viewBox='0 0 24 24' fill='none' stroke='%238a94a6' stroke-width='2.5'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 9px center;
    cursor: pointer;
    transition: border-color .15s;
  }}

  input[type=text] {{
    background-image: none;
    padding-right: 10px;
    cursor: text;
  }}

  select:focus, input:focus {{
    border-color: #3B7DD8;
  }}

  .filter-grid {{
    display: grid;
    grid-template-columns: 1.2fr 1fr 1.2fr;
    gap: 10px 16px;
  }}

  .table-scroll {{
    overflow-y: auto;
    height: 470px;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
  }}

  thead tr {{
    background: #f8fafc;
    position: sticky;
    top: 0;
    z-index: 2;
  }}

  th {{
    padding: 10px 10px;
    text-align: left;
    color: #8a94a6;
    font-weight: 700;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: .04em;
    border-bottom: 2px solid #eef0f5;
    white-space: nowrap;
  }}

  td {{
    padding: 10px 10px;
    height: 46px;
    border-bottom: 1px solid #f5f7fa;
    color: #2d3748;
    vertical-align: middle;
    white-space: nowrap;
  }}

  tbody tr:hover td {{
    background: #fafbfd;
  }}

  .table-footer {{
    padding: 11px 16px;
    border-top: 2px solid #eef0f5;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.78rem;
    color: #8a94a6;
    background: white;
  }}

  .btn-download {{
    display: flex;
    align-items: center;
    gap: 5px;
    background: white;
    border: 1.5px solid #e2e8f0;
    border-radius: 8px;
    padding: 7px 12px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.76rem;
    font-weight: 700;
    color: #1a2235;
    cursor: pointer;
    transition: all .15s;
  }}

  .btn-download:hover {{
    border-color: #3B7DD8;
    color: #3B7DD8;
  }}
</style>
</head>
<body>

<div class="card">
  <div class="card-header">
    <span style="font-size:1.08rem">🔍</span>
    Search &amp; Filters
  </div>
  <div class="card-body">
    <div class="filter-grid">
      <div class="form-group">
        <label>Search by Student ID or Name</label>
        <input type="text" id="searchInput" placeholder="STU-" oninput="applyFilters()"/>
      </div>
      <div class="form-group">
        <label>Risk Level <span class="tip">?</span></label>
        <select id="riskFilter" onchange="applyFilters()">
          <option value="All">All</option>
          <option value="High">High</option>
          <option value="Medium">Medium</option>
          <option value="Low">Low</option>
        </select>
      </div>
      <div class="form-group">
        <label>Recommended Action <span class="tip">?</span></label>
        <select id="actionFilter" onchange="applyFilters()">
          <option value="All">All</option>
          {action_options}
        </select>
      </div>
    </div>
  </div>
</div>

<div class="card">
  <div class="card-header">
    <span style="color:#3B7DD8;font-size:1.08rem">📊</span>
    Student Table
  </div>

  <div class="table-scroll">
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Name</th>
          <th style="text-align:center">G1</th>
          <th style="text-align:center">G2</th>
          <th style="text-align:center">Absences</th>
          <th style="text-align:center">Failures</th>
          <th style="text-align:center">Risk Score</th>
          <th>Risk Level</th>
          <th>Recommended Action</th>
        </tr>
      </thead>
      <tbody id="tableBody">
        {rows_html}
      </tbody>
    </table>
  </div>

  <div class="table-footer">
    <span>Total Students Displayed: <b id="totalCount" style="color:#1a2235">{total}</b></span>
    <button class="btn-download" onclick="downloadCSV()">⬇ Download Filtered Student List</button>
  </div>
</div>

<script>
const allRows = Array.from(document.querySelectorAll('#tableBody tr'));

function applyFilters() {{
  const search = document.getElementById('searchInput').value.toLowerCase().trim();
  const riskSelect = document.getElementById('riskFilter');
  const actionSelect = document.getElementById('actionFilter');

  let currentRisk = riskSelect.value;
  let currentAction = actionSelect.value;
  let visible = 0;

  const exactMatches = allRows.filter(tr => {{
    const id = (tr.dataset.id || '').toLowerCase().trim();
    return search && id === search;
  }});

  if (exactMatches.length === 1) {{
    riskSelect.value = exactMatches[0].dataset.level || 'All';
    actionSelect.value = exactMatches[0].dataset.action || 'All';
    currentRisk = riskSelect.value;
    currentAction = actionSelect.value;
  }}

  if (!search) {{
    riskSelect.value = 'All';
    actionSelect.value = 'All';
    currentRisk = 'All';
    currentAction = 'All';
  }}

  allRows.forEach(tr => {{
    const id = (tr.dataset.id || '').toLowerCase();
    const name = (tr.dataset.name || '').toLowerCase();
    const lvl = (tr.dataset.level || '');
    const act = (tr.dataset.action || '');

    const matchSearch = !search || id.includes(search) || name.includes(search);
    const matchRisk = currentRisk === 'All' || lvl === currentRisk;
    const matchAction = currentAction === 'All' || act === currentAction;
    const show = matchSearch && matchRisk && matchAction;

    tr.style.display = show ? '' : 'none';
    if (show) visible++;
  }});

  document.getElementById('totalCount').textContent = visible;
}}

function downloadCSV() {{
  const visible = allRows.filter(r => r.style.display !== 'none');
  const hdrs = ['ID','Name','G1','G2','Absences','Failures','Risk Score','Risk Level','Recommended Action'];
  let csv = hdrs.join(',') + '\\n';

  visible.forEach(tr => {{
    const cells = Array.from(tr.querySelectorAll('td'))
      .map(td => '"' + td.textContent.trim().replace(/"/g, '""') + '"');
    csv += cells.join(',') + '\\n';
  }});

  const a = Object.assign(document.createElement('a'), {{
    href: URL.createObjectURL(new Blob([csv], {{type:'text/csv'}})),
    download: 'filtered_students.csv'
  }});
  a.click();
}}
</script>

</body>
</html>
        """, height=IFRAME_H, scrolling=False)

    else:
        components.html(f"""
        {SHARED_CSS}
        <div style="background:white;border:2px dashed #c8d4e8;border-radius:14px;
                    padding:60px;text-align:center;margin-top:10px">
            <div style="font-size:2.5rem;margin-bottom:10px">📂</div>
            <div style="font-size:1rem;font-weight:600;color:#4a5568;margin-bottom:6px">
                No Student Data Available
            </div>
            <div style="font-size:0.85rem;color:#8a94a6">
                Upload a CSV file from the sidebar to view students
            </div>
        </div>
        """, height=240)

elif st.session_state.page == "Risk Analysis":

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import io
    import base64
    import numpy as np

    # ── helper: fig → base64 png ──────────────────────────────
    def fig_to_b64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                    facecolor="white")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    # ── shared style for every chart ─────────────────────────
    def style_ax(ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=11, fontweight="700",
                     color="#1a2235", pad=10, loc="left")
        ax.set_xlabel(xlabel, fontsize=8.5, color="#8a94a6", labelpad=6)
        ax.set_ylabel(ylabel, fontsize=8.5, color="#8a94a6", labelpad=6)
        ax.tick_params(colors="#8a94a6", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#eef0f5")
        ax.spines["bottom"].set_color("#eef0f5")
        ax.set_facecolor("white")
        ax.yaxis.grid(True, color="#f0f4f8", linewidth=0.8)
        ax.set_axisbelow(True)

    # ── page header ──────────────────────────────────────────
    st.markdown("""
    <div style="background:white;border-radius:12px;padding:14px 22px;margin-bottom:20px;
                box-shadow:0 2px 8px rgba(0,0,0,0.06);display:flex;align-items:center;gap:10px">
        <span style="font-size:1.4rem">🎓</span>
        <div>
            <div style="font-size:1.15rem;font-weight:800;color:#1a2235">
                Academic Risk Analysis
            </div>
            <div style="font-size:0.78rem;color:#8a94a6;margin-top:1px">
                Visual insights into student performance and risk drivers
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)
        if "G3" in df.columns:
            df = df.drop(columns=["G3"])
        df = predict_if_needed(df, model)

        # Use fixed thresholds so natural distribution shows (Low/High dominate)
        df["Risk Level"] = df["Risk Probability"].apply(
            lambda x: "Low" if x < 0.30 else ("Medium" if x < 0.60 else "High")
        )

        df["Risk Score"] = df["Risk Probability"]

        # ── card CSS ─────────────────────────────────────────
        st.markdown("""
        <style>
        .chart-card {
            background: white;
            border-radius: 14px;
            border: 1px solid #eef2f7;
            box-shadow: 0 2px 12px rgba(15,23,42,0.06);
            padding: 18px 20px 14px;
            margin-bottom: 2px;
        }
        .chart-card img { width:100%; border-radius:6px; }
        .heatmap-card {
            background: white;
            border-radius: 14px;
            border: 1px solid #eef2f7;
            box-shadow: 0 2px 12px rgba(15,23,42,0.06);
            padding: 20px 22px 18px;
            margin-top: 8px;
        }
        .heatmap-card img { width:100%; border-radius:6px; }
        </style>
        """, unsafe_allow_html=True)

        # =================================================
        # ROW 1 — Risk Level Distribution + Risk Gauge
        # =================================================
        col1, col2 = st.columns(2, gap="medium")

        # ── Chart 1: Risk Level Distribution ─────────────
        with col1:
            risk_counts = df["Risk Level"].value_counts().reindex(
                ["Low", "Medium", "High"], fill_value=0
            )
            bar_colors = ["#4e8ef7", "#f5a623", "#e74c3c"]

            fig, ax = plt.subplots(figsize=(5, 3.2))
            fig.patch.set_facecolor("white")
            bars = ax.bar(
                risk_counts.index, risk_counts.values,
                color=bar_colors, width=0.5, zorder=3
            )
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    str(int(bar.get_height())),
                    ha="center", va="bottom",
                    fontsize=8, color="#1a2235", fontweight="600"
                )
            style_ax(ax, "Risk Level Distribution", "Risk Level", "Students")
            plt.tight_layout(pad=0.8)
            b64 = fig_to_b64(fig)
            plt.close(fig)

            st.markdown(f"""
            <div class="chart-card">
                <img src="data:image/png;base64,{b64}"/>
            </div>""", unsafe_allow_html=True)

        # ── Chart 2: Overall Risk Gauge / Speedometer ─────
        with col2:
            avg_score = float(df["Risk Score"].mean())

            # needle angle: 0.0 → left (0°)  1.0 → right (180°)
            angle = avg_score * 180.0
            rad   = angle * 3.14159265 / 180.0

            zone_colors = ["#27ae60", "#f5a623", "#e67e22", "#e74c3c"]
            zone_labels = ["Low", "Moderate", "Elevated", "High"]
            zone_starts = [0, 45, 90, 135]
            zone_widths = [45, 45, 45, 45]

            fig, ax = plt.subplots(figsize=(5, 3.2))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            for start, width, color in zip(zone_starts, zone_widths, zone_colors):
                theta = np.linspace(
                    np.radians(start), np.radians(start + width), 120
                )
                r_outer, r_inner = 1.0, 0.62
                xs = (
                    list(np.cos(theta) * r_outer)+ list(np.cos(theta[::-1]) * r_inner)
                )
                ys = (
                    list(np.sin(theta) * r_outer)+ list(np.sin(theta[::-1]) * r_inner)
                )
                ax.fill(xs, ys, color=color, alpha=0.88, zorder=2)

            for start in zone_starts[1:]:
                a = np.radians(start)
                ax.plot(
                    [np.cos(a) * 0.60, np.cos(a) * 1.02],
                    [np.sin(a) * 0.60, np.sin(a) * 1.02],
                    color="white", linewidth=2.2, zorder=3
                )

            nx = np.cos(rad) * 0.78
            ny = np.sin(rad) * 0.78
            ax.annotate(
                "", xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#1a2235",
                    lw=2.2,
                    mutation_scale=14,
                ),
                zorder=5,
            )
            ax.plot(0, 0, "o", color="#1a2235", markersize=7, zorder=6)

            ax.text(
                0, -0.22, f"{avg_score:.2f}",
                ha="center", va="center",
                fontsize=18, fontweight="800", color="#1a2235", zorder=6
            )
            ax.text(
                0, -0.40, "Average Risk Score",
                ha="center", va="center",
                fontsize=7.5, color="#8a94a6", zorder=6
            )

            label_positions = [22, 67, 112, 157]
            for pos, lbl, col in zip(label_positions, zone_labels, zone_colors):
                a = np.radians(pos)
                ax.text(
                    np.cos(a) * 1.16, np.sin(a) * 1.16,
                    lbl,
                    ha="center", va="center",
                    fontsize=6.5, fontweight="700", color=col
                )

            ax.text(
                0, 1.28, "Overall Risk Gauge",
                ha="center", va="center",
                fontsize=11, fontweight="700", color="#1a2235"
            )

            ax.set_xlim(-1.55, 1.55)
            ax.set_ylim(-0.70, 1.55)
            ax.axis("off")
            plt.tight_layout(pad=0.3)
            b64 = fig_to_b64(fig)
            plt.close(fig)

            st.markdown(f"""
            <div class="chart-card">
                <img src="data:image/png;base64,{b64}"/>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

        # =================================================
        # ROW 2 — Absences Boxplot + Failures Bar
        # =================================================
        col3, col4 = st.columns(2, gap="medium")

        # ── Chart 3: Absences vs Risk Level ───────────────
        with col3:
            if "absences" in df.columns:
                order   = ["Low", "Medium", "High"]
                palette = {"Low": "#4e8ef7", "Medium": "#f5a623", "High": "#e74c3c"}

                fig, ax = plt.subplots(figsize=(5, 3.2))
                fig.patch.set_facecolor("white")
                sns.boxplot(
                    x="Risk Level", y="absences", data=df,
                    order=order, palette=palette,
                    width=0.45, linewidth=1.2,
                    flierprops=dict(
                        marker="o", markersize=3,
                        markerfacecolor="#8a94a6",
                        markeredgecolor="none"
                    ),
                    ax=ax, zorder=3
                )
                style_ax(ax, "Absences vs Risk Level", "Risk Level", "Absences")
                plt.tight_layout(pad=0.8)
                b64 = fig_to_b64(fig)
                plt.close(fig)

                st.markdown(f"""
                <div class="chart-card">
                    <img src="data:image/png;base64,{b64}"/>
                </div>""", unsafe_allow_html=True)

        # ── Chart 4: Past Failures vs Risk Score ──────────
        with col4:
            if "failures" in df.columns:
                failure_risk = df.groupby("failures")["Risk Score"].mean()
                max_idx      = failure_risk.idxmax()
                bar_cols     = [
                    "#f5a623" if i == max_idx else "#7eb8f7"
                    for i in failure_risk.index
                ]

                fig, ax = plt.subplots(figsize=(5, 3.2))
                fig.patch.set_facecolor("white")
                bars = ax.bar(
                    failure_risk.index.astype(str),
                    failure_risk.values,
                    color=bar_cols, width=0.5, zorder=3
                )
                for bar in bars:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{bar.get_height():.2f}",
                        ha="center", va="bottom",
                        fontsize=7.5, color="#1a2235", fontweight="600"
                    )
                style_ax(
                    ax, "Past Failures vs Risk Score",
                    "Number of Failures", "Average Risk Score"
                )
                plt.tight_layout(pad=0.8)
                b64 = fig_to_b64(fig)
                plt.close(fig)

                st.markdown(f"""
                <div class="chart-card">
                    <img src="data:image/png;base64,{b64}"/>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

        # =================================================
        # ROW 3 — Feature Importance Chart (full width)
        # =================================================
        if hasattr(model, "feature_importances_"):
            features    = model.feature_names_in_
            importances = model.feature_importances_

            # sort descending
            sorted_idx  = importances.argsort()[::-1]
            sorted_feat = [features[i] for i in sorted_idx]
            sorted_imp  = importances[sorted_idx]

            # colour: top feature = orange, rest = blue gradient
            fi_colors = [
                "#f5a623" if i == 0 else "#4e8ef7"
                for i in range(len(sorted_feat))
            ]

            fig, ax = plt.subplots(figsize=(10, 3.4))
            fig.patch.set_facecolor("white")
            bars = ax.bar(
                sorted_feat, sorted_imp,
                color=fi_colors, width=0.5, zorder=3
            )
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom",
                    fontsize=7.5, color="#1a2235", fontweight="600"
                )

            style_ax(
                ax, "Feature Importance — What Drives Risk the Most?",
                "Feature", "Importance Score"
            )
            plt.tight_layout(pad=0.8)
            b64 = fig_to_b64(fig)
            plt.close(fig)

            st.markdown(f"""
            <div class="heatmap-card">
                <img src="data:image/png;base64,{b64}"/>
            </div>""", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

        # =================================================
        # FEATURE CORRELATION HEATMAP
        # =================================================
        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

        heatmap_cols = [
            c for c in
            ["absences", "failures", "studytime", "G1", "G2",
             "Risk Score", "internet", "higher"]
            if c in df.columns
        ]

        if len(heatmap_cols) >= 3:
            corr = df[heatmap_cols].corr().round(2)

            n = len(heatmap_cols)
            fig, ax = plt.subplots(figsize=(n * 1.05 + 1.5, n * 0.72 + 1.2))
            fig.patch.set_facecolor("white")

            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                center=0,
                vmin=-1,
                vmax=1,
                linewidths=0.5,
                linecolor="#eef2f7",
                annot_kws={"size": 8.5, "weight": "600", "color": "#1a2235"},
                square=True,
                ax=ax,
                cbar_kws={"shrink": 0.75, "ticks": [1, 0.5, 0, -0.5, -1]},
            )

            ax.set_title(
                "Feature Correlation Heatmap",
                fontsize=12, fontweight="800",
                color="#1a2235", pad=14, loc="left"
            )
            ax.tick_params(
                axis="x", labelsize=8.5, colors="#4a5568",
                rotation=0, bottom=False
            )
            ax.tick_params(
                axis="y", labelsize=8.5, colors="#4a5568",
                rotation=0, left=False
            )
            ax.set_facecolor("white")

            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=8, colors="#8a94a6")
            cbar.outline.set_edgecolor("#eef2f7")

            plt.tight_layout(pad=0.8)
            b64 = fig_to_b64(fig)
            plt.close(fig)

            st.markdown(f"""
            <div class="heatmap-card">
                <img src="data:image/png;base64,{b64}"/>
            </div>""", unsafe_allow_html=True)

    else:
        components.html(f"""
        {SHARED_CSS}
        <div style="background:white;border:2px dashed #c8d4e8;border-radius:14px;
                    padding:60px;text-align:center;margin-top:10px">
            <div style="font-size:2.5rem;margin-bottom:10px">📂</div>
            <div style="font-size:1rem;font-weight:600;color:#4a5568;margin-bottom:6px">
                No Data Available
            </div>
            <div style="font-size:0.85rem;color:#8a94a6">
                Upload a CSV file from the sidebar to view risk analysis
            </div>
        </div>
        """, height=220)

elif st.session_state.page == "Reports":

    # ── page header (left-aligned, not centered) ─────────────
    st.markdown("""
    <div style="background:white;border-radius:12px;padding:14px 22px;margin-bottom:20px;
                box-shadow:0 2px 8px rgba(0,0,0,0.06);display:flex;align-items:center;gap:10px">
        <span style="font-size:1.4rem">📑</span>
        <div>
            <div style="font-size:1.15rem;font-weight:800;color:#1a2235">
                Academic Risk Report
            </div>
            <div style="font-size:0.78rem;color:#8a94a6;margin-top:1px">
                Executive summary and intervention planning
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file, sep=None, engine="python")

        if "G3" in df.columns:
            df = df.drop(columns=["G3"])

        df = predict_if_needed(df, model)

        # ── generate Student ID and Name if not present ───────
        if "Student ID" not in df.columns:
            if "ID" in df.columns:
                df["Student ID"] = df["ID"]
            elif "student_id" in df.columns:
                df["Student ID"] = df["student_id"]
            elif "id" in df.columns:
                df["Student ID"] = df["id"]
            else:
                df["Student ID"] = [f"STU-{i+1:03d}" for i in df.index]

        if "Name" not in df.columns:
            if "name" in df.columns:
                df["Name"] = df["name"]
            else:
                df["Name"] = [f"Student #{i+1}" for i in df.index]

        # ── statistics ───────────────────────────────────────
        total_students = len(df)
        low_count  = int((df["Risk Level"] == "Low").sum())
        med_count  = int((df["Risk Level"] == "Medium").sum())
        high_count = int((df["Risk Level"] == "High").sum())
        avg_risk   = float(df["Risk Probability"].mean())
        low_pct    = low_count  / total_students * 100
        med_pct    = med_count  / total_students * 100
        high_pct   = high_count / total_students * 100

        # ── intervention counts ───────────────────────────────
        intervention_counts = (
            df["Recommended Action"].value_counts().to_dict()
        )

        # ── high risk table rows ──────────────────────────────
        high_risk_df = df[df["Risk Level"] == "High"].copy()
        important_cols = [
            c for c in
            ["Risk Probability", "Risk Level", "Recommended Action",
             "absences", "failures", "studytime"]
            if c in high_risk_df.columns
        ]
        high_risk_sorted = (
            high_risk_df[important_cols]
            .sort_values("Risk Probability", ascending=False)
        )

        def hr_rows(sub_df):
            out = ""
            for _, row in sub_df.iterrows():
                prob = float(row["Risk Probability"])
                out += f"""
                <tr>
                  <td style='font-weight:700;color:#e74c3c'>{prob:.2f}</td>
                  <td><span style='background:#fde8e8;color:#e74c3c;padding:2px 10px;
                      border-radius:20px;font-size:0.74rem;font-weight:600'>High</span></td>
                  <td style='color:#555;font-size:0.8rem'>{row.get("Recommended Action","")}</td>
                  <td style='text-align:center'>{int(row.get("absences",0))}</td>
                  <td style='text-align:center'>{int(row.get("failures",0))}</td>
                  <td style='text-align:center'>{int(row.get("studytime",0))}</td>
                </tr>"""
            return out

        hr_html = hr_rows(high_risk_sorted)

        # ── intervention table rows ───────────────────────────
        def iv_rows():
            out = ""
            for action, count in intervention_counts.items():
                out += f"""
                <tr>
                  <td style='color:#2d3748;font-size:0.82rem'>{action}</td>
                  <td style='text-align:right;font-weight:700;color:#1a2235'>{count}</td>
                </tr>"""
            return out

        iv_html = iv_rows()

        # ── alert banner ──────────────────────────────────────
        if high_pct > 25:
            alert_color  = "#fde8e8"
            alert_border = "#e74c3c"
            alert_icon   = "🚨"
            alert_text   = (
                f"ALERT: {high_pct:.1f}% of students are HIGH RISK. "
                "Immediate intervention recommended."
            )
        elif high_pct > 15:
            alert_color  = "#fff8e1"
            alert_border = "#f5a623"
            alert_icon   = "⚠️"
            alert_text   = (
                f"Warning: {high_pct:.1f}% of students are at high academic risk."
            )
        else:
            alert_color  = "#e8f5e9"
            alert_border = "#27ae60"
            alert_icon   = "✅"
            alert_text   = "Academic risk levels are currently stable."

        # ── progress bar widths ───────────────────────────────
        low_w  = f"{low_pct:.1f}%"
        med_w  = f"{med_pct:.1f}%"
        high_w = f"{high_pct:.1f}%"

        # ── CSVs for download buttons ─────────────────────────
        import io as _io
        import base64 as _b64

        def to_b64_csv(dataframe):
            buf = _io.StringIO()
            dataframe.to_csv(buf, index=False)
            return _b64.b64encode(buf.getvalue().encode()).decode()

        full_b64 = to_b64_csv(df)
        high_b64 = to_b64_csv(high_risk_df)

        # Intervention plan — always includes Student ID + Name
        id_col   = next(
            (c for c in ["Student ID", "student_id", "ID", "id"]
             if c in df.columns), None
        )
        name_col = next(
            (c for c in ["Name", "name"] if c in df.columns), None
        )
        final_plan_cols = []
        if id_col:
            final_plan_cols.append(id_col)
        if name_col:
            final_plan_cols.append(name_col)
        final_plan_cols += [
            c for c in
            ["Risk Level", "Risk Probability", "Recommended Action"]
            if c in df.columns
        ]
        plan_b64 = to_b64_csv(df[final_plan_cols])

        # ── render everything via components.html ─────────────
        import streamlit.components.v1 as _comp

        _comp.html(f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{
    font-family:'DM Sans',sans-serif;
    background:#f0f4f8;
    padding:0 2px 16px;
    font-size:13px;
    color:#2d3748;
  }}

  /* ── KPI cards ── */
  .kpi-row {{
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:12px;
    margin-bottom:12px;
  }}
  .kpi-card {{
    background:white;
    border-radius:12px;
    border:1px solid #eef2f7;
    box-shadow:0 2px 8px rgba(15,23,42,0.05);
    padding:16px 18px;
  }}
  .kpi-label {{
    font-size:0.72rem;
    font-weight:600;
    color:#8a94a6;
    text-transform:uppercase;
    letter-spacing:.04em;
    margin-bottom:6px;
  }}
  .kpi-value {{
    font-size:1.7rem;
    font-weight:800;
    color:#1a2235;
    line-height:1;
  }}

  /* ── alert banner ── */
  .alert-banner {{
    background:{alert_color};
    border:1.5px solid {alert_border};
    border-radius:10px;
    padding:11px 16px;
    font-size:0.82rem;
    font-weight:600;
    color:#1a2235;
    margin-bottom:12px;
  }}

  /* ── two-col row ── */
  .two-col {{
    display:grid;
    grid-template-columns:1.4fr 1fr;
    gap:12px;
    margin-bottom:12px;
  }}

  /* ── white card ── */
  .card {{
    background:white;
    border-radius:12px;
    border:1px solid #eef2f7;
    box-shadow:0 2px 8px rgba(15,23,42,0.05);
    padding:18px 20px;
  }}
  .card-title {{
    font-size:0.88rem;
    font-weight:800;
    color:#1a2235;
    margin-bottom:14px;
  }}

  /* ── executive summary text ── */
  .summary-text {{
    font-size:0.82rem;
    line-height:1.75;
    color:#4a5568;
  }}
  .summary-text b {{ color:#1a2235; }}
  .summary-text i {{ color:#e74c3c; font-style:normal; font-weight:600; }}

  /* ── intervention table ── */
  .iv-table {{ width:100%; border-collapse:collapse; }}
  .iv-table th {{
    font-size:0.7rem;
    font-weight:700;
    color:#8a94a6;
    text-transform:uppercase;
    letter-spacing:.04em;
    border-bottom:2px solid #eef0f5;
    padding:6px 8px;
    text-align:left;
  }}
  .iv-table td {{
    padding:9px 8px;
    border-bottom:1px solid #f5f7fa;
    vertical-align:middle;
  }}
  .iv-table tbody tr:hover td {{ background:#fafbfd; }}

  /* ── distribution bars ── */
  .dist-row {{
    display:flex;
    align-items:center;
    gap:10px;
    margin-bottom:10px;
  }}
  .dist-label {{
    width:90px;
    font-size:0.78rem;
    font-weight:600;
    color:#4a5568;
    flex-shrink:0;
  }}
  .dist-track {{
    flex:1;
    background:#f0f4f8;
    border-radius:20px;
    height:12px;
    overflow:hidden;
  }}
  .dist-fill {{
    height:100%;
    border-radius:20px;
  }}
  .dist-pct {{
    width:42px;
    font-size:0.78rem;
    font-weight:700;
    text-align:right;
    flex-shrink:0;
  }}

  /* ── full-width card ── */
  .full-card {{
    background:white;
    border-radius:12px;
    border:1px solid #eef2f7;
    box-shadow:0 2px 8px rgba(15,23,42,0.05);
    padding:18px 20px;
    margin-bottom:12px;
    overflow-x:auto;
  }}

  /* ── high risk table ── */
  .hr-table {{ width:100%; border-collapse:collapse; }}
  .hr-table th {{
    font-size:0.68rem;
    font-weight:700;
    color:#8a94a6;
    text-transform:uppercase;
    letter-spacing:.04em;
    border-bottom:2px solid #eef0f5;
    padding:8px 10px;
    text-align:left;
    white-space:nowrap;
    background:#f8fafc;
  }}
  .hr-table td {{
    padding:9px 10px;
    border-bottom:1px solid #f5f7fa;
    vertical-align:middle;
    white-space:nowrap;
  }}
  .hr-table tbody tr:hover td {{ background:#fafbfd; }}

  /* ── download buttons ── */
  .btn-row {{
    display:flex;
    gap:10px;
    margin-bottom:12px;
  }}
  .btn {{
    flex:1;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    gap:6px;
    padding:10px 14px;
    border-radius:10px;
    font-family:'DM Sans',sans-serif;
    font-size:0.8rem;
    font-weight:700;
    cursor:pointer;
    border:none;
    text-decoration:none;
    color:white;
  }}
  .btn-blue   {{ background:#2f80ed; }}
  .btn-orange {{ background:#f5a623; }}
  .btn-green  {{ background:#27ae60; }}
  .btn:hover  {{ opacity:0.88; }}

  /* ── bottom two-col ── */
  .bottom-row {{
    display:grid;
    grid-template-columns:1fr 1.4fr;
    gap:12px;
  }}
  .ethical-card {{
    background:white;
    border-radius:12px;
    border:1.5px solid #f5a623;
    padding:16px 18px;
  }}
  .sysinfo-card {{
    background:white;
    border-radius:12px;
    border:1px solid #eef2f7;
    box-shadow:0 2px 8px rgba(15,23,42,0.05);
    padding:16px 18px;
  }}
  .dot {{
    width:8px; height:8px; border-radius:50%;
    display:inline-block; margin-right:6px;
    vertical-align:middle;
    flex-shrink:0;
  }}
  .sysinfo-row {{
    font-size:0.8rem;
    color:#4a5568;
    margin-bottom:7px;
    display:flex;
    align-items:center;
  }}
  .sysinfo-row b {{ color:#1a2235; margin-left:4px; }}
</style>
</head>
<body>

<!-- KPI CARDS -->
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-label">Total Students</div>
    <div class="kpi-value">{total_students}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">High Risk Students</div>
    <div class="kpi-value" style="color:#e74c3c">{high_count}
      <span style="font-size:0.9rem;font-weight:600;color:#e74c3c">&nbsp;{high_pct:.1f}%</span>
    </div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Low Risk Students</div>
    <div class="kpi-value" style="color:#27ae60">{low_count}
      <span style="font-size:0.9rem;font-weight:600;color:#27ae60">&nbsp;{low_pct:.1f}%</span>
    </div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Average Risk Score</div>
    <div class="kpi-value">{avg_risk:.2f}</div>
  </div>
</div>

<!-- ALERT BANNER -->
<div class="alert-banner">
  {alert_icon} {alert_text}
</div>

<!-- EXECUTIVE SUMMARY + INTERVENTION -->
<div class="two-col">
  <div class="card">
    <div class="card-title">Executive Summary</div>
    <div class="summary-text">
      This report analyzes <b>{total_students} students</b> using the AcademicGuard model.<br><br>
      <b>{high_count} students ({high_pct:.1f}%)</b> are classified as <i>High Risk</i>.<br><br>
      Additionally, <b>{med_count} students ({med_pct:.1f}%)</b> are identified as
      <span style="color:#f5a623;font-weight:600">Medium Risk</span>,
      while <b>{low_count} students ({low_pct:.1f}%)</b> are categorized as
      <span style="color:#27ae60;font-weight:600">Low Risk</span>.<br><br>
      The average academic risk score is <b>{avg_risk:.2f}</b>,
      indicating an overall moderate level of risk.
    </div>
  </div>
  <div class="card">
    <div class="card-title">Intervention Action Breakdown</div>
    <table class="iv-table">
      <thead>
        <tr>
          <th>Intervention Type</th>
          <th style="text-align:right">Students</th>
        </tr>
      </thead>
      <tbody>
        {iv_html}
      </tbody>
    </table>
  </div>
</div>

<!-- RISK LEVEL DISTRIBUTION -->
<div class="full-card">
  <div class="card-title">Risk Level Distribution</div>
  <div class="dist-row">
    <div class="dist-label">Low Risk</div>
    <div class="dist-track">
      <div class="dist-fill" style="width:{low_w};background:#4e8ef7"></div>
    </div>
    <div class="dist-pct" style="color:#4e8ef7">{low_pct:.1f}%</div>
  </div>
  <div class="dist-row">
    <div class="dist-label">Medium Risk</div>
    <div class="dist-track">
      <div class="dist-fill" style="width:{med_w};background:#f5a623"></div>
    </div>
    <div class="dist-pct" style="color:#f5a623">{med_pct:.1f}%</div>
  </div>
  <div class="dist-row">
    <div class="dist-label">High Risk</div>
    <div class="dist-track">
      <div class="dist-fill" style="width:{high_w};background:#e74c3c"></div>
    </div>
    <div class="dist-pct" style="color:#e74c3c">{high_pct:.1f}%</div>
  </div>
</div>

<!-- HIGH RISK TABLE -->
<div class="full-card">
  <div class="card-title">High Risk Students Requiring Attention</div>
  <div style="overflow-y:auto;height:410px;border-radius:6px;">
    <table class="hr-table">
      <thead style="position:sticky;top:0;z-index:2;">
        <tr>
          <th>Risk Probability</th>
          <th>Risk Level</th>
          <th>Recommended Action</th>
          <th style="text-align:center">Absences</th>
          <th style="text-align:center">Failures</th>
          <th style="text-align:center">Studytime</th>
        </tr>
      </thead>
      <tbody>
        {hr_html}
      </tbody>
    </table>
  </div>
</div>

<!-- DOWNLOAD BUTTONS -->
<div class="btn-row">
  <a class="btn btn-blue"
     href="data:text/csv;base64,{full_b64}"
     download="academic_risk_report.csv">
    📥 Full Report CSV
  </a>
  <a class="btn btn-orange"
     href="data:text/csv;base64,{high_b64}"
     download="high_risk_students.csv">
    ⚠️ High Risk Students CSV
  </a>
  <a class="btn btn-green"
     href="data:text/csv;base64,{plan_b64}"
     download="intervention_plan.csv">
    🩺 Intervention Plan CSV
  </a>
</div>

<!-- ETHICAL NOTICE + SYSTEM INFO -->
<div class="bottom-row">
  <div class="ethical-card">
    <div class="card-title" style="color:#f5a623">⚠️ Ethical Notice</div>
    <div style="font-size:0.78rem;color:#4a5568;line-height:1.7;font-style:italic">
      Predictions are estimates and should not be the sole basis for decisions.
      Always consider the full context of a student's situation.
    </div>
  </div>
  <div class="sysinfo-card">
    <div class="card-title">ℹ️ System Information</div>
    <div class="sysinfo-row">
      <span class="dot" style="background:#2f80ed"></span>
      <span>System: <b>AcademicGuard</b></span>
    </div>
    <div class="sysinfo-row">
      <span class="dot" style="background:#27ae60"></span>
      <span>Model: <b>Random Forest Classifier</b></span>
    </div>
    <div class="sysinfo-row">
      <span class="dot" style="background:#f5a623"></span>
      <span>Total Students Analyzed: <b>{total_students}</b></span>
    </div>
    <div class="sysinfo-row">
      <span class="dot" style="background:#e74c3c"></span>
      <span>Dashboard Version: <b>1.0</b></span>
    </div>
  </div>
</div>

</body>
</html>
        """, height=1480, scrolling=True)

    else:
        st.markdown("""
        <div style="background:white;border:2px dashed #c8d4e8;border-radius:14px;
                    padding:60px;text-align:center;margin-top:10px">
            <div style="font-size:2.5rem;margin-bottom:10px">📂</div>
            <div style="font-size:1rem;font-weight:600;color:#4a5568;margin-bottom:6px">
                No Data Available
            </div>
            <div style="font-size:0.85rem;color:#8a94a6">
                Upload a CSV file from the sidebar to generate reports
            </div>
        </div>
        """, unsafe_allow_html=True)
