###############################################################
# Nutri-Bridge — Weekly Meal Planner with Pantry Mode
# + REAL-TIME Kaggle Dataset Fetch (no file uploads needed)
###############################################################

import streamlit as st
import pandas as pd
import numpy as np
from ast import literal_eval
from collections import Counter
from pulp import LpProblem, LpVariable, LpMinimize, LpBinary, lpSum, PULP_CBC_CMD
import matplotlib.pyplot as plt
import kagglehub
import os

st.set_page_config(page_title="Nutri-Bridge Planner", layout="wide")
st.title("🥗 Nutri-Bridge — Weekly Nutrition Planner")


###############################################################
# 🔐 Kaggle API Credentials (Put your key here OR inside .streamlit/secrets.toml)
###############################################################

# If running locally – put kaggle.json in ~/.kaggle/
os.environ["KAGGLE_USERNAME"] = "maharajaharan"
os.environ["KAGGLE_KEY"] = "98008443e2a71475ee2fe25424da561c"


###############################################################
# 🔄 REAL-TIME KAGGLE DATASET FETCHER
###############################################################

@st.cache_data(show_spinner=True)
def load_kaggle_realtime():
    st.info("🔄 Fetching dataset directly from Kaggle... Please wait...")

    df = kagglehub.load_dataset(
        kagglehub.KaggleDatasetAdapter.PANDAS,
        "maharajaharan/raw-recipes-small",
        "RAW_recipes_small.csv"
    )

    st.success("✅ Dataset loaded live from Kaggle!")
    return df


df = load_kaggle_realtime()


###############################################################
# 🧹 PREPROCESSING
###############################################################

def safe_list(v):
    try:
        val = literal_eval(v)
        return val if isinstance(val, (list, tuple)) else []
    except:
        return []

def normalize_ing(x):
    x = x.lower().strip().replace("-", " ").replace(",", "")
    words = x.split()
    words = [w[:-1] if w.endswith("s") else w for w in words]
    return "_".join(words)

def detect_meal(name, tags):
    t = (str(name) + str(tags)).lower()
    if "breakfast" in t: return "breakfast"
    if "lunch" in t: return "lunch"
    if "dinner" in t: return "dinner"
    return "lunch"

def detect_cuisine(name, tags):
    cuisines = ["indian","italian","mexican","chinese","thai","american"]
    t = (str(name)+str(tags)).lower()
    for c in cuisines:
        if c in t: return c
    return "general"


# Apply preprocessing
df["ingredients"] = df["ingredients"].apply(safe_list)
df["steps"] = df["steps"].apply(safe_list)
df["ing_list"] = df["ingredients"].apply(lambda lst: [normalize_ing(i) for i in lst])

df["calories"] = df["nutrition"].apply(lambda x: literal_eval(x)[0] if isinstance(x,str) else 0)
df["protein_g"] = df["nutrition"].apply(lambda x: literal_eval(x)[1] if isinstance(x,str) else 0)

df["meal_type"] = df.apply(lambda r: detect_meal(r["name"], r["tags"]), axis=1)
df["cuisine"] = df.apply(lambda r: detect_cuisine(r["name"], r["tags"]), axis=1)
df["cost_est"] = 20


###############################################################
# 🧍 USER INPUTS
###############################################################

st.sidebar.header("User Profile")

age = st.sidebar.number_input("Age", 10, 80, 25)
gender = st.sidebar.selectbox("Gender", ["male","female"])
height = st.sidebar.number_input("Height (cm)", 120, 210, 170)
weight = st.sidebar.number_input("Weight (kg)", 30, 200, 70)
activity = st.sidebar.selectbox("Activity Level", ["sedentary","light","moderate","active"])

preferred_cuisines = st.sidebar.text_input(
    "Preferred cuisines", value="indian,italian"
).lower().replace(" ", "").split(",")

restrictions = [normalize_ing(r) for r in st.sidebar.text_input(
    "Dietary restrictions", value="sugar"
).split(",")]

# Pantry Mode 🧺
pantry_items = st.sidebar.text_area(
    "Pantry ingredients (comma separated)",
    placeholder="onion, rice, pasta"
)
pantry_items = {normalize_ing(i) for i in pantry_items.split(",") if i.strip()}

cal_tol = st.sidebar.slider("Calorie tolerance ±%", 0.05, 0.30, 0.12)
pro_tol = st.sidebar.slider("Protein tolerance ±%", 0.05, 0.40, 0.15)


def calorie_target():
    base = 10*weight + 6.25*height - 5*age + (5 if gender=="male" else -161)
    factor = {"sedentary":1.2,"light":1.375,"moderate":1.55,"active":1.75}
    return int(base * factor[activity])

def protein_target():
    return int(weight * 0.9)


def pantry_score(ingredients):
    return sum(1 for i in ingredients if i in pantry_items)


###############################################################
# ILP DAY PLANNER
###############################################################

def plan_day(day, df_filtered, used):
    pool = df_filtered.sample(min(300, len(df_filtered)))

    CAL = calorie_target()
    PRO = protein_target()

    rec_map = {(r["meal_type"], int(r["id"])): r for _, r in pool.iterrows()}

    prob = LpProblem(f"Day_{day}", LpMinimize)
    x = {k: LpVariable(f"x_{k}", 0,1,LpBinary) for k in rec_map}

    for m in ["breakfast","lunch","dinner"]:
        prob += lpSum(x[k] for k in x if k[0]==m) == 1

    prob += lpSum(x[k]*rec_map[k]["calories"] for k in x) >= CAL*(1-cal_tol)
    prob += lpSum(x[k]*rec_map[k]["calories"] for k in x) <= CAL*(1+cal_tol)
    prob += lpSum(x[k]*rec_map[k]["protein_g"] for k in x) >= PRO*(1-pro_tol)

    # ⭐ Pantry Mode Reward
    prob += lpSum(x[k] * pantry_score(rec_map[k]["ing_list"]) for k in x)

    prob.solve(PULP_CBC_CMD(msg=0))

    chosen = []
    for k,v in x.items():
        if v.value() == 1:
            chosen.append(rec_map[k])
            used.add(rec_map[k]["id"])

    return chosen, used


###############################################################
# 🚀 GENERATE WEEKLY PLAN
###############################################################

if st.button("GENERATE WEEKLY PLAN 🚀"):

    used_recipes = set()
    week = []
    summary = []
    shopping = Counter()

    MEAL_ORDER = {"breakfast":1, "lunch":2, "dinner":3}

    for d in range(1, 8):
        df_f = df[
            df["cuisine"].isin(preferred_cuisines) &
            ~df["ing_list"].apply(lambda lst: any(r in lst for r in restrictions))
        ]

        meals, used_recipes = plan_day(d, df_f, used_recipes)

        # 🔥 FIXED: Sort meals in correct order
        meals = sorted(meals, key=lambda r: MEAL_ORDER[r["meal_type"]])

        total_cal = 0
        total_pro = 0

        for r in meals:
            week.append([
                d,
                r["meal_type"],
                r["cuisine"],
                r["name"],
                ", ".join(r["ingredients"]),
                " → ".join(r["steps"]),
                f"{r['calories']} cal",
                f"{r['protein_g']} g"
            ])

            total_cal += r["calories"]
            total_pro += r["protein_g"]

            for ing in r["ing_list"]:
                shopping[ing] += 1

        summary.append([d, round(total_cal), round(total_pro)])

    week_df = pd.DataFrame(week, columns=[
        "Day","Meal","Cuisine","Recipe","Ingredients","Steps","Calories","Protein"
    ])

    st.subheader("📅 Weekly Meal Plan")
    st.dataframe(week_df)

    st.download_button("⬇ Download Weekly Plan CSV",
                       week_df.to_csv(index=False), "weekly_plan.csv")

    summary_df = pd.DataFrame(summary, columns=["Day","Calories","Protein"])
    st.subheader("📊 Daily Nutrition Summary")
    st.dataframe(summary_df)


    ###############################################################
    # 🛒 Shopping List
    ###############################################################

    COST_MAP = {"produce":5,"dairy":10,"meat":30,"pantry":10,"other":8}

    def classify(ing):
        ing = ing.lower()
        if any(x in ing for x in ["onion","pepper","ginger","carrot"]): return "produce"
        if "milk" in ing or "cheese" in ing: return "dairy"
        if any(x in ing for x in ["chicken","beef","fish","pork"]): return "meat"
        if any(x in ing for x in ["rice","salt","oil","pasta"]): return "pantry"
        return "other"

    rows = []
    total_cost = 0

    for ing, qty in shopping.items():
        cat = classify(ing)
        cost = COST_MAP[cat] * qty
        rows.append([cat.capitalize(), ing, qty, cost])
        total_cost += cost

    shop_df = pd.DataFrame(rows, columns=["Category","Ingredient","Qty","Estimated Cost"])
    st.subheader("🛒 Categorized Shopping List")
    st.dataframe(shop_df)

    st.success(f"Total Estimated Cost: ₹ {total_cost}")

    st.download_button("⬇ Download Shopping List CSV",
                       shop_df.to_csv(index=False),
                       "shopping_list.csv")


    ###############################################################
    # 📈 GRAPHS
    ###############################################################

    st.subheader("📈 Calories & Protein Trend")

    plt.style.use("default")
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # --- Calories Line ---
    ax1.plot(
        summary_df["Day"],
        summary_df["Calories"],
        marker='o',
        linewidth=3,
        color="#4F81BD",
        markerfacecolor="white",
        markeredgewidth=2
    )
    ax1.set_xlabel("Day", fontsize=12)
    ax1.set_ylabel("Calories (kcal)", color="#4F81BD", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#4F81BD")

    # --- Protein Line ---
    ax2 = ax1.twinx()
    ax2.plot(
        summary_df["Day"],
        summary_df["Protein"],
        marker='s',
        linewidth=3,
        linestyle="--",
        color="#9BBB59",
        markerfacecolor="white",
        markeredgewidth=2
    )
    ax2.set_ylabel("Protein (g)", color="#9BBB59", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#9BBB59")

    # --- Grid & Layout ---
    ax1.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()

    # --- Combined Legend ---
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, ["Calories", "Protein"], loc="upper left")
    st.pyplot(fig)

    st.subheader("🥘 Top Ingredients Used")

    freq = (
        pd.DataFrame(shopping.items(), columns=["Ingredient", "Qty"])
            .sort_values("Qty", ascending=False)
            .head(12)
    )

    fig2, ax = plt.subplots(figsize=(9, 6))

    bars = ax.barh(
        freq["Ingredient"],
        freq["Qty"],
        color="#5DADE2",
        edgecolor="black",
        alpha=0.85
    )

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{int(width)}",
                va='center', fontsize=10)

    ax.set_xlabel("Frequency", fontsize=12)
    ax.set_title("Most Frequently Used Ingredients", fontsize=14, weight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig2.tight_layout()
    st.pyplot(fig2)


