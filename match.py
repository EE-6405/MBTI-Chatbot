import pandas as pd

# Load data
profile_df = pd.read_csv("./data/MBTI_Full_Profile.csv", encoding="utf-8-sig", index_col="mbti")
reasons_df = pd.read_csv("./data/MBTI_Match_Reasons_Database.csv", encoding="utf-8-sig", index_col="mbti")
compat_df = pd.read_csv("./data/MBTI_128.csv", encoding="utf-8-sig")

# Normalize MBTI strings
profile_df.index = profile_df.index.str.strip().str.upper()
reasons_df.index = reasons_df.index.str.strip().str.upper()
compat_df["original_mbti"] = compat_df["original_mbti"].str.upper()

# Best match rules
BEST_MATCHES = {
    'INFP': ['INFJ','ENFJ'], 'ENFP': ['INFJ','ENFJ'],
    'INFJ': ['ENFP','ENFJ'], 'ENFJ': ['INFP','INFJ'],
    'INTJ': ['ENTJ','ENTP'], 'ENTJ': ['INFJ','ENFJ'],
    'INTP': ['ENTP','INFJ'], 'ENTP': ['INFJ','ENFJ'],
    'ISFP': ['ESFJ','ESTJ'], 'ESFP': ['INFP','ENFP'],
    'ISTP': ['ESFJ','ESTJ'], 'ESTP': ['INFJ','INTJ'],
    'ISFJ': ['ENFP','ESFJ'], 'ESFJ': ['INFP','ISFJ'],
    'ISTJ': ['ENFP','INTP'], 'ESTJ': ['INFJ','ESFJ']
}
# Function 1: Get profile details
def get_mbti_profile(mbti):
    mbti = mbti.strip().upper()
    if mbti not in profile_df.index:
        return None
    row = profile_df.loc[mbti]
    return {
        "analysis": row["analysis"] if "analysis" in row else row.get("description", ""),
        "strengths": row["strengths"],
        "recommended_careers": row["recommended_careers"],
        "career_reasons": row["career_reasons"]
    }

# Function 2: Recommend best match
def recommend_partner(mbti):
    mbti = mbti.strip().upper()
    matches = BEST_MATCHES.get(mbti, [])
    candidates = compat_df[compat_df["original_mbti"].isin(matches)]
    if candidates.empty:
        return None
    best = candidates.loc[candidates["total_score"].idxmax()]
    return {
        "person_id": int(best["person_id"]),
        "mbti": best["original_mbti"]
    }

# Function 3: Get match reason
def get_match_reason(user_mbti, partner_mbti):
    user_mbti = user_mbti.strip().upper()
    partner_mbti = partner_mbti.strip().upper()
    if user_mbti not in reasons_df.index:
        return ""
    row = reasons_df.loc[user_mbti]
    if row["partner_1"] == partner_mbti:
        return row["reason_1"]
    elif row["partner_2"] == partner_mbti:
        return row["reason_2"]
    return ""

# Main function (only added this part)
def match_main(test_mbti):
    '''
    result = {
        "profile": get_mbti_profile(test_mbti),
        "partner": recommend_partner(test_mbti)
    }
    if result["partner"]:
        result["reason"] = get_match_reason(test_mbti, result["partner"]["mbti"])
    '''
    profile = get_mbti_profile(test_mbti)
    partner = recommend_partner(test_mbti)
    reason = get_match_reason(test_mbti, partner["mbti"]) if partner else ""

    if not profile:
        return f"Sorry, we couldn't find information for MBTI type '{test_mbti}'."

    result_str = f"Your MBTI type is {test_mbti.upper()}.\n\n"
    result_str += "Profile Summary:\n"
    result_str += f"- Analysis: {profile['analysis']}\n"
    result_str += f"- Strengths: {profile['strengths']}\n"
    result_str += f"- Recommended Careers: {profile['recommended_careers']}\n"
    result_str += f"- Career Reason: {profile['career_reasons']}\n\n"

    if partner:
        result_str += f"Your Best Match is: {partner['mbti']} (Person ID: {partner['person_id']})\n"
        result_str += f"Reason for Match: {reason}\n"
    else:
        result_str += "Sorry, we couldn't find a recommended match.\n"

    print(result_str)
    return result_str

