import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from supabase import create_client, Client


# ---------------- Supabase 설정 ----------------

@st.cache_resource
def get_supabase_client() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


# ---------------- 유틸 함수 ----------------

def get_kst_now():
    return datetime.utcnow() + timedelta(hours=9)


def empty_data_df():
    return pd.DataFrame(
        columns=[
            "id", "date", "weight", "status",
            "calories_breakdown", "total_calories",
            "score", "total_score"
        ]
    )


# ---------------- DB load/save ----------------

def load_data():
    supabase = get_supabase_client()
    resp = supabase.table("mom_records").select("*").execute()
    data = resp.data or []

    if not data:
        return empty_data_df()

    df = pd.DataFrame(data)

    for col in [
        "id","date","weight","status","calories_breakdown",
        "total_calories","score","total_score"
    ]:
        if col not in df.columns:
            df[col] = None

    df = df.sort_values("date").reset_index(drop=True)
    return df


def save_data(df: pd.DataFrame):
    supabase = get_supabase_client()

    supabase.table("mom_records").delete().neq("id", 0).execute()

    if df.empty:
        return

    records = df.copy()
    if "id" in records.columns:
        records = records.drop(columns=["id"])

    data_list = records.to_dict(orient="records")
    supabase.table("mom_records").insert(data_list).execute()


def load_config():
    supabase = get_supabase_client()
    resp = supabase.table("mom_config").select("*").execute()
    rows = resp.data or []

    config = {}
    for row in rows:
        key = row.get("key")
        value = row.get("value")
        if key is None:
            continue
        config[key] = value
    return config


def save_config(config: dict):
    supabase = get_supabase_client()
    supabase.table("mom_config").delete().neq("id", 0).execute()

    rows = [{"key": k, "value": str(v)} for k, v in config.items()]
    if rows:
        supabase.table("mom_config").insert(rows).execute()


# ---------------- 로직 ----------------

def find_last_T(df):
    t_rows = df[df["status"] == "T"].copy()
    if t_rows.empty:
        return None
    t_rows = t_rows.sort_values("date")
    row = t_rows.iloc[-1]
    return row["date"], row["weight"]


def mark_F(df, last_t_date, today_date):
    last = datetime.strptime(last_t_date, "%Y-%m-%d")
    today = datetime.strptime(today_date, "%Y-%m-%d")

    missing_days = (today - last).days - 1
    if missing_days <= 0:
        return df, 0

    existing_dates = set(df["date"].tolist())
    added_f = 0

    for i in range(1, missing_days + 1):
        f_date = (last + timedelta(days=i)).strftime("%Y-%m-%d")
        if f_date in existing_dates:
            continue

        row = {
            "id": None, "date": f_date, "weight": None,
            "status": "F", "calories_breakdown": {},
            "total_calories": None, "score": 0.0, "total_score": None
        }

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        added_f += 1

    return df, added_f


def calculate_score(prev_weight, today_weight, num_f):
    score = 0.0

    if prev_weight is not None:
        diff = today_weight - prev_weight
        step = int(abs(diff) / 0.1)
        if diff < 0:
            score += step * 0.2
        elif diff > 0:
            score -= step * 0.2

    score -= num_f * 0.3
    return round(score, 2)


def recalc_total_scores(df):
    total = 0.0
    for i in df.index:
        score = df.loc[i, "score"]
        if pd.isna(score):
            df.loc[i, "total_score"] = total
            continue
        total += float(score)
        df.loc[i, "total_score"] = round(total, 2)
    return df


# ---------------- A 화면 ----------------

def page_A():
    st.title("성진 다이어트 프로그램 – A 화면 (기록/포인트)  (엄마용)")

    df = load_data()

    now_kst = get_kst_now()
    today_iso = now_kst.strftime("%Y-%m-%d")
    today_kr = now_kst.strftime("%Y년 %m월 %d일")

    today_row = df[df["date"] == today_iso]

    if not today_row.empty:
        prev_cb = today_row.iloc[0]["calories_breakdown"]
        if not isinstance(prev_cb, dict):
            prev_cb = {"아침": 0, "점심": 0, "저녁": 0, "간식": 0}
        prev_weight = today_row.iloc[0]["weight"]
    else:
        prev_cb = {"아침": 0, "점심": 0, "저녁": 0, "간식": 0}
        prev_weight = None

    st.subheader("오늘 날짜")
    st.write(f"한국 기준: **{today_kr}**")

    last_T = find_last_T(df)
    if last_T:
        last_t_date, last_t_weight = last_T
        st.info(f"직전 T: {last_t_date} / {last_t_weight} kg")
    else:
        last_t_date, last_t_weight = None, None
        st.info("직전 T 없음")

    st.markdown("---")
    st.subheader("오늘 몸무게 / 식단 입력")

    weight = st.number_input(
        "오늘 몸무게 (kg)", min_value=30.0, max_value=300.0,
        step=0.1, value=float(prev_weight) if prev_weight else 60.0,
        format="%.1f"
    )

    col1, col2 = st.columns(2)

    with col1:
        bf = st.text_input("아침 식단")
        lu = st.text_input("점심 식단")
        di = st.text_input("저녁 식단")
        sn = st.text_input("간식")

    with col2:
        kcal_bf = st.number_input("아침 칼로리", min_value=0, step=10, value=int(prev_cb.get("아침", 0)))
        kcal_lu = st.number_input("점심 칼로리", min_value=0, step=10, value=int(prev_cb.get("점심", 0)))
        kcal_di = st.number_input("저녁 칼로리", min_value=0, step=10, value=int(prev_cb.get("저녁", 0)))
        kcal_sn = st.number_input("간식 칼로리", min_value=0, step=10, value=int(prev_cb.get("간식", 0)))

    total_kcal = kcal_bf + kcal_lu + kcal_di + kcal_sn
    st.write(f"**총합 칼로리:** {total_kcal} kcal")

    # ---------------- T 기록 저장 ----------------
    if st.button("오늘 T 기록 저장"):

        cutoff_date = (
            datetime.strptime(today_iso, "%Y-%m-%d") - timedelta(days=30)
        ).strftime("%Y-%m-%d")
        df = df[df["date"] >= cutoff_date].copy()

        if not df.empty and ((df["date"] == today_iso) & (df["status"] == "T")).any():
            st.error("오늘 날짜에 이미 T가 있습니다.")
            return

        if last_t_date:
            df, num_f = mark_F(df, last_t_date, today_iso)
        else:
            num_f = 0

        today_score = calculate_score(last_t_weight, weight, num_f)

        if df.empty:
            prev_total = 0.0
        else:
            prev_total = df["total_score"].fillna(0).astype(float).iloc[-1]

        today_total = round(prev_total + today_score, 2)

        cb_dict = {
            "아침": kcal_bf,
            "점심": kcal_lu,
            "저녁": kcal_di,
            "간식": kcal_sn,
        }

        new_row = {
            "id": None,
            "date": today_iso,
            "weight": float(weight),
            "status": "T",
            "calories_breakdown": cb_dict,
            "total_calories": int(total_kcal),
            "score": float(today_score),
            "total_score": float(today_total),
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        save_data(df)

        st.success("오늘 T 기록 저장 완료!")
        st.write(f"F 개수: {num_f}")
        st.write(f"오늘 포인트: **{today_score}점**")
        st.write(f"총합 포인트: **{today_total}점**")

    if st.button("B 화면으로 이동"):
        st.session_state["page"] = "B"
        st.rerun()

    st.markdown("---")
    st.subheader("T 기록 수정하기")

    t_rows = df[df["status"] == "T"].sort_values("date")
    t_dates = t_rows["date"].tolist()

    if not t_dates:
        st.write("수정할 T 기록 없음.")
        return

    selected_date = st.selectbox("수정할 날짜", t_dates)

    row = t_rows[t_rows["date"] == selected_date].iloc[0]
    old_weight = row["weight"]
    old_cb = row["calories_breakdown"] or {}
    if not isinstance(old_cb, dict):
        old_cb = {"아침": 0, "점심": 0, "저녁": 0, "간식": 0}
    old_total = row["total_calories"]

    st.write(f"기존 몸무게: **{old_weight} kg** / 총칼로리 **{old_total} kcal**")

    new_weight = st.number_input(
        "새 몸무게", min_value=30.0, max_value=300.0,
        step=0.1, value=float(old_weight)
    )

    st.write("식단 칼로리 수정")
    new_kcal_bf = st.number_input("아침",  min_value=0, step=10, value=int(old_cb.get("아침", 0)))
    new_kcal_lu = st.number_input("점심",  min_value=0, step=10, value=int(old_cb.get("점심", 0)))
    new_kcal_di = st.number_input("저녁",  min_value=0, step=10, value=int(old_cb.get("저녁", 0)))
    new_kcal_sn = st.number_input("간식",  min_value=0, step=10, value=int(old_cb.get("간식", 0)))

    new_total = new_kcal_bf + new_kcal_lu + new_kcal_di + new_kcal_sn
    new_cb_dict = {
        "아침": new_kcal_bf,
        "점심": new_kcal_lu,
        "저녁": new_kcal_di,
        "간식": new_kcal_sn,
    }

    if st.button("수정 저장"):
        df.loc[df["date"] == selected_date, "weight"] = float(new_weight)
        df.loc[df["date"] == selected_date, "calories_breakdown"] = new_cb_dict
        df.loc[df["date"] == selected_date, "total_calories"] = int(new_total)

        before_df = df[df["status"] == "T"].sort_values("date")
        idx = before_df[before_df["date"] == selected_date].index[0]
        prev_pos = list(before_df.index).index(idx) - 1

        if prev_pos >= 0:
            prev_weight_val = float(before_df.iloc[prev_pos]["weight"])
        else:
            prev_weight_val = None

        df.loc[df["date"] == selected_date, "score"] = calculate_score(
            prev_weight_val, float(new_weight), 0
        )

        df = df.sort_values("date")
        df = recalc_total_scores(df)

        save_data(df)
        st.success("수정 완료!")


# ---------------- B 화면 ----------------

def page_B():
    st.title("성진 다이어트 프로그램 – B 화면 (그래프/키)  (엄마용)")

    df = load_data()
    config = load_config()

    st.subheader("키 설정")

    current_raw = config.get("height_cm", "160.0")
    try:
        current_height = float(current_raw)
    except:
        current_height = 160.0

    height = st.number_input(
        "키 (cm)",
        min_value=100.0, max_value=250.0,
        step=0.1, value=float(current_height),
        format="%.1f"
    )

    if st.button("키 저장"):
        config["height_cm"] = float(height)
        save_config(config)
        st.success("저장됨.")

    st.markdown("---")
    st.subheader("최근 30일 기록")

    if df.empty:
        st.write("기록 없음.")
        return

    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_dt").reset_index(drop=True)
    recent = df.tail(30).copy()

    display = recent.copy()
    display["w"] = display["weight"].apply(lambda x: "-" if pd.isna(x) else x)

    st.dataframe(
        display[["date","w","status","total_calories","score","total_score"]].rename(
            columns={
                "date": "날짜", "w": "몸무게", "status": "T/F",
                "total_calories": "칼로리", "score": "오늘점수", "total_score": "총합점수"
            }
        ),
        use_container_width=True
    )

    st.subheader("체중 그래프 (T만)")

    graph_df = recent[recent["status"] == "T"].copy()
    if graph_df.empty:
        st.write("그래프 표시할 T가 없습니다.")
        return

    graph_df["weight"] = graph_df["weight"].astype(float)
    graph_df = graph_df.set_index("date_dt")
    st.line_chart(graph_df["weight"])

    if st.button("A 화면으로 이동"):
        st.session_state["page"] = "A"
        st.rerun()


# ---------------- 메인 ----------------

def main():
    st.set_page_config(page_title="성진 다이어트 프로그램 (엄마용)", layout="wide")

    if "page" not in st.session_state:
        st.session_state["page"] = "A"

    if st.session_state["page"] == "A":
        page_A()
    else:
        page_B()


if __name__ == "__main__":
    main()
