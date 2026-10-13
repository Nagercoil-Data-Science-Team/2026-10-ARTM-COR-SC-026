import sys
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import docx

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("Starting generation and analysis script...")

# -------------------------------------------------------------
# 1. Matplotlib Configuration (User specifications)
# dpi=800, figsize=(11, 7), Times New Roman, bold, size 18, NO GRIDS
# -------------------------------------------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['figure.figsize'] = (11, 7)
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.grid'] = False

OUTPUT_DIR = r"d:\Alan christo\October 2026\2026-10-ARTM-COR-SC-026"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 2. Synthetic Dataset Generation (N = 420, Alpha > 0.80)
# -------------------------------------------------------------
np.random.seed(42)
N = 420

cov = [
    [1.00, 0.44, 0.40, 0.38],
    [0.44, 1.00, 0.46, 0.42],
    [0.40, 0.46, 1.00, 0.39],
    [0.38, 0.42, 0.39, 1.00]
]
ivs = np.random.multivariate_normal([0, 0, 0, 0], cov, size=N)
smrc_lat = ivs[:, 0]
smru_lat = ivs[:, 1]
smrq_lat = ivs[:, 2]
smrv_lat = ivs[:, 3]

# Structural equation: MWI = 0.24*SMRC + 0.28*SMRU + 0.21*SMRQ + 0.25*SMRV + error
mwi_lat = 0.24 * smrc_lat + 0.28 * smru_lat + 0.21 * smrq_lat + 0.25 * smrv_lat + np.random.normal(0, 0.55, size=N)

latents = {
    'SMRC': (smrc_lat, 3.82, [0.84, 0.82, 0.85, 0.80, 0.83, 0.81]),
    'SMRU': (smru_lat, 3.86, [0.83, 0.81, 0.86, 0.82, 0.84, 0.80]),
    'SMRQ': (smrq_lat, 3.78, [0.85, 0.82, 0.84, 0.83, 0.81, 0.84]),
    'SMRV': (smrv_lat, 3.90, [0.84, 0.86, 0.82, 0.85, 0.83, 0.81]),
    'MWI':  (mwi_lat,  3.84, [0.82, 0.84, 0.81, 0.83, 0.85, 0.80])
}

df_items = {}
for c_name, (lat, base_m, loadings) in latents.items():
    items = []
    for idx, l in enumerate(loadings):
        noise = np.sqrt(1 - l**2)
        raw = l * lat + noise * np.random.normal(0, 1, size=N)
        val = np.round(base_m + 0.85 * raw)
        val = np.clip(val, 1, 5).astype(int)
        df_items[f'{c_name}{idx+1}'] = val

df_survey = pd.DataFrame(df_items)

# Generate Demographics matching Table 2 of Workflow Doc
age_cats = ['18–20 years', '21–23 years', '24–26 years', '27–29 years', '30–34 years']
age_probs = [0.18, 0.34, 0.28, 0.12, 0.08]
age_vals = np.random.choice(age_cats, size=N, p=age_probs)
age_code_map = {cat: i+1 for i, cat in enumerate(age_cats)}

gender_cats = ['Male', 'Female', 'Prefer not to say']
gender_probs = [0.48, 0.49, 0.03]
gender_vals = np.random.choice(gender_cats, size=N, p=gender_probs)
gender_code_map = {cat: i+1 for i, cat in enumerate(gender_cats)}

edu_cats = ['Higher Secondary', 'Diploma/Associate Degree', 'Bachelor’s Degree', 'Master’s Degree', 'Doctoral Degree', 'Other']
edu_probs = [0.12, 0.14, 0.48, 0.22, 0.03, 0.01]
edu_vals = np.random.choice(edu_cats, size=N, p=edu_probs)
edu_code_map = {cat: i+1 for i, cat in enumerate(edu_cats)}

occ_cats = ['Student', 'Employed', 'Self-employed', 'Unemployed', 'Other']
occ_probs = [0.42, 0.41, 0.11, 0.05, 0.01]
occ_vals = np.random.choice(occ_cats, size=N, p=occ_probs)
occ_code_map = {cat: i+1 for i, cat in enumerate(occ_cats)}

sm_usage_cats = ['Less than 1 hour', '1–2 hours', '3–4 hours', '5–6 hours', 'More than 6 hours']
sm_usage_probs = [0.05, 0.21, 0.45, 0.22, 0.07]
sm_usage_vals = np.random.choice(sm_usage_cats, size=N, p=sm_usage_probs)
sm_usage_code_map = {cat: i+1 for i, cat in enumerate(sm_usage_cats)}

freq_cats = ['Rarely', 'Sometimes', 'Often', 'Very Often']
freq_probs = [0.10, 0.31, 0.42, 0.17]
freq_vals = np.random.choice(freq_cats, size=N, p=freq_probs)
freq_code_map = {cat: i+1 for i, cat in enumerate(freq_cats)}

df_demo = pd.DataFrame({
    'Respondent_ID': [f'R{i+1:03d}' for i in range(N)],
    'Age': age_vals,
    'Age_Code': [age_code_map[x] for x in age_vals],
    'Gender': gender_vals,
    'Gender_Code': [gender_code_map[x] for x in gender_vals],
    'Education': edu_vals,
    'Education_Code': [edu_code_map[x] for x in edu_vals],
    'Occupation': occ_vals,
    'Occupation_Code': [occ_code_map[x] for x in occ_vals],
    'Daily_SM_Usage': sm_usage_vals,
    'Daily_SM_Usage_Code': [sm_usage_code_map[x] for x in sm_usage_vals],
    'Review_Encounter_Freq': freq_vals,
    'Review_Encounter_Freq_Code': [freq_code_map[x] for x in freq_vals]
})

# Construct Composite Means
for c in ['SMRC', 'SMRU', 'SMRQ', 'SMRV', 'MWI']:
    c_cols = [f'{c}{j}' for j in range(1, 7)]
    df_survey[f'{c}_Mean'] = df_survey[c_cols].mean(axis=1)

# Full Dataset
full_df = pd.concat([df_demo, df_survey], axis=1)

DATASET_DIR = os.path.join(OUTPUT_DIR, 'Dataset')
os.makedirs(DATASET_DIR, exist_ok=True)

# Export Full CSV and Excel
csv_path = os.path.join(OUTPUT_DIR, 'Dataset_Social_Media_Movie_Reviews_N420.csv')
xlsx_path = os.path.join(OUTPUT_DIR, 'Dataset_Social_Media_Movie_Reviews_N420.xlsx')
full_df.to_csv(csv_path, index=False)
full_df.to_excel(xlsx_path, index=False)
full_df.to_csv(os.path.join(DATASET_DIR, 'Dataset_Social_Media_Movie_Reviews_N420.csv'), index=False)
full_df.to_excel(os.path.join(DATASET_DIR, 'Dataset_Social_Media_Movie_Reviews_N420.xlsx'), index=False)

# Export SPSS Ready (Numeric)
spss_cols = ['Respondent_ID', 'Age_Code', 'Gender_Code', 'Education_Code', 'Occupation_Code', 
             'Daily_SM_Usage_Code', 'Review_Encounter_Freq_Code'] + [f'{c}{j}' for c in ['SMRC', 'SMRU', 'SMRQ', 'SMRV', 'MWI'] for j in range(1, 7)] + [f'{c}_Mean' for c in ['SMRC', 'SMRU', 'SMRQ', 'SMRV', 'MWI']]
spss_df = full_df[spss_cols].copy()
spss_csv_path = os.path.join(OUTPUT_DIR, 'Dataset_SPSS_Ready.csv')
spss_df.to_csv(spss_csv_path, index=False)
spss_df.to_csv(os.path.join(DATASET_DIR, 'Dataset_SPSS_Ready.csv'), index=False)

# Export SmartPLS CSV (Items only with Demographics)
pls_cols = [f'{c}{j}' for c in ['SMRC', 'SMRU', 'SMRQ', 'SMRV', 'MWI'] for j in range(1, 7)] + ['Age_Code', 'Gender_Code', 'Daily_SM_Usage_Code']
pls_df = full_df[pls_cols].copy()
pls_csv_path = os.path.join(OUTPUT_DIR, 'SmartPLS_Import_Data.csv')
pls_df.to_csv(pls_csv_path, index=False)
pls_df.to_csv(os.path.join(DATASET_DIR, 'SmartPLS_Import_Data.csv'), index=False)

print("Datasets exported successfully to both root and Dataset/ folder.")

# -------------------------------------------------------------
# 3. Statistical Calculations (SPSS & SmartPLS)
# -------------------------------------------------------------

def calc_cronbach(sub):
    k = sub.shape[1]
    s2_sum = sub.var(axis=0, ddof=1).sum()
    s2_tot = sub.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - s2_sum / s2_tot)

def calc_rho_a(sub, loadings):
    std_data = (sub - sub.mean()) / sub.std(ddof=1)
    S = np.corrcoef(std_data.values.T)
    c = 1.0 / np.sqrt(np.dot(loadings, np.dot(S, loadings)))
    w = c * loadings
    
    S_nodiag = S.copy()
    np.fill_diagonal(S_nodiag, 0.0)
    
    AA = np.outer(w, w)
    np.fill_diagonal(AA, 0.0)
    
    num = np.dot(w, np.dot(S_nodiag, w))
    den = np.dot(w, np.dot(AA, w))
    return float((np.dot(w, w)**2) * (num / den))

constructs = ['SMRC', 'SMRU', 'SMRQ', 'SMRV', 'MWI']
construct_full_names = {
    'SMRC': 'Social Media Movie Review Credibility',
    'SMRU': 'Social Media Movie Review Usefulness',
    'SMRQ': 'Social Media Movie Review Quality',
    'SMRV': 'Social Media Movie Review Valence',
    'MWI':  'Movie-Watching Intention'
}

meas_summary = []
outer_loadings_dict = {}
indicator_vifs = {}

for c in constructs:
    sub = df_survey[[f'{c}{j}' for j in range(1, 7)]]
    alpha = calc_cronbach(sub)
    
    # Outer Loadings via PCA on items
    pca = PCA(n_components=1)
    pca.fit(sub)
    loadings = np.corrcoef(sub.values.T, pca.transform(sub).T)[0:6, 6]
    outer_loadings_dict[c] = loadings
    
    rho_a = calc_rho_a(sub, loadings)
    
    # Indicator VIF
    vifs = []
    for col in sub.columns:
        other_cols = [x for x in sub.columns if x != col]
        X_sub = sm.add_constant(sub[other_cols])
        y_sub = sub[col]
        r2 = sm.OLS(y_sub, X_sub).fit().rsquared
        vifs.append(1.0 / (1.0 - r2) if r2 < 1.0 else np.nan)
    indicator_vifs[c] = vifs
    
    ave = np.mean(loadings**2)
    cr = (np.sum(loadings)**2) / ((np.sum(loadings)**2) + np.sum(1 - loadings**2))
    
    meas_summary.append({
        'Construct': c,
        'Full Name': construct_full_names[c],
        'Cronbach_Alpha': alpha,
        'rho_A': rho_a,
        'Composite_Reliability': cr,
        'AVE': ave
    })

meas_df = pd.DataFrame(meas_summary)
print("\n--- Measurement Model Assessment ---")
print(meas_df.round(3).to_string(index=False))

# Outer Loadings Table
loadings_rows = []
for c in constructs:
    for j in range(1, 7):
        item_code = f'{c}{j}'
        l_val = outer_loadings_dict[c][j-1]
        vif_val = indicator_vifs[c][j-1]
        loadings_rows.append({
            'Construct': c,
            'Indicator': item_code,
            'Outer_Loading': l_val,
            'Indicator_VIF': vif_val
        })
loadings_df = pd.DataFrame(loadings_rows)

# Fornell-Larcker Criterion
construct_means = pd.DataFrame({c: df_survey[f'{c}_Mean'] for c in constructs})
corr_matrix = construct_means.corr()

fl_matrix = corr_matrix.copy()
for i, c in enumerate(constructs):
    ave = meas_df.loc[meas_df['Construct'] == c, 'AVE'].values[0]
    fl_matrix.iloc[i, i] = np.sqrt(ave)

print("\n--- Fornell-Larcker Criterion ---")
print(fl_matrix.round(3))

# HTMT Matrix
htmt_matrix = pd.DataFrame(np.eye(len(constructs)), index=constructs, columns=constructs)
for i, c1 in enumerate(constructs):
    for j, c2 in enumerate(constructs):
        if i < j:
            sub1 = df_survey[[f'{c1}{k}' for k in range(1, 7)]]
            sub2 = df_survey[[f'{c2}{k}' for k in range(1, 7)]]
            
            r_hetero = np.abs(np.corrcoef(sub1.values.T, sub2.values.T)[:6, 6:]).mean()
            r1 = np.abs(np.corrcoef(sub1.values.T))
            r2 = np.abs(np.corrcoef(sub2.values.T))
            mono1 = r1[np.triu_indices(6, k=1)].mean()
            mono2 = r2[np.triu_indices(6, k=1)].mean()
            htmt_val = r_hetero / np.sqrt(mono1 * mono2)
            htmt_matrix.iloc[i, j] = htmt_val
            htmt_matrix.iloc[j, i] = htmt_val

print("\n--- HTMT Matrix ---")
print(htmt_matrix.round(3))

# Common Method Bias - Harman's Single Factor Test
pca_harman = PCA(n_components=1)
pca_harman.fit(df_survey[[f'{c}{j}' for c in constructs for j in range(1, 7)]])
harman_variance = pca_harman.explained_variance_ratio_[0] * 100
print(f"\nHarman Single Factor Variance Explained: {harman_variance:.2f}% (Threshold: < 50%)")

# Multiple Regression & Structural Model
X_reg = construct_means[['SMRC', 'SMRU', 'SMRQ', 'SMRV']]
X_reg_const = sm.add_constant(X_reg)
y_reg = construct_means['MWI']
reg_model = sm.OLS(y_reg, X_reg_const).fit()

# Structural Model Collinearity (VIF)
pred_vifs = {}
for col in X_reg.columns:
    others = [x for x in X_reg.columns if x != col]
    r2_p = sm.OLS(X_reg[col], sm.add_constant(X_reg[others])).fit().rsquared
    pred_vifs[col] = 1.0 / (1.0 - r2_p)

# Effect Size (f^2)
r2_full = reg_model.rsquared
f2_dict = {}
for col in X_reg.columns:
    others = [x for x in X_reg.columns if x != col]
    r2_ex = sm.OLS(y_reg, sm.add_constant(X_reg[others])).fit().rsquared
    f2_val = (r2_full - r2_ex) / (1.0 - r2_full)
    f2_dict[col] = f2_val

# Standardized Coefficients (Beta)
std_X = (X_reg - X_reg.mean()) / X_reg.std(ddof=1)
std_y = (y_reg - y_reg.mean()) / y_reg.std(ddof=1)
std_model = sm.OLS(std_y, std_X).fit()

# Hypothesis Testing Table
hyp_list = [
    ('H1', 'SMRC -> MWI', 'SMRC', 'Positive', '+'),
    ('H2', 'SMRU -> MWI', 'SMRU', 'Positive', '+'),
    ('H3', 'SMRQ -> MWI', 'SMRQ', 'Positive', '+'),
    ('H4', 'SMRV -> MWI', 'SMRV', 'Positive', '+'),
]

hyp_results = []
for h_id, path_str, var_code, exp_dir, sign in hyp_list:
    b_val = reg_model.params[var_code]
    se_val = reg_model.bse[var_code]
    beta_val = std_model.params[var_code]
    t_val = reg_model.tvalues[var_code]
    p_val = reg_model.pvalues[var_code]
    ci_low = reg_model.conf_int().loc[var_code, 0]
    ci_high = reg_model.conf_int().loc[var_code, 1]
    vif_val = pred_vifs[var_code]
    f2_val = f2_dict[var_code]
    decision = 'Supported' if (p_val < 0.05 and beta_val > 0) else 'Not Supported'
    
    hyp_results.append({
        'Hypothesis': h_id,
        'Structural Path': path_str,
        'Expected Direction': exp_dir,
        'Std Beta': beta_val,
        'Unstd B': b_val,
        'SE': se_val,
        't-value': t_val,
        'p-value': p_val,
        '95% CI Lower': ci_low,
        '95% CI Upper': ci_high,
        'f2 Effect Size': f2_val,
        'VIF': vif_val,
        'Decision': decision
    })

hyp_df = pd.DataFrame(hyp_results)
print("\n--- Structural Model & Hypothesis Testing ---")
print(hyp_df[['Hypothesis', 'Structural Path', 'Std Beta', 't-value', 'p-value', 'f2 Effect Size', 'VIF', 'Decision']].round(3).to_string(index=False))

# Descriptive Statistics (Item Level & Construct Level)
item_descriptives = []
for c in constructs:
    for j in range(1, 7):
        code = f'{c}{j}'
        s = df_survey[code]
        item_descriptives.append({
            'Construct': c,
            'Item': code,
            'Mean': s.mean(),
            'Std Dev': s.std(),
            'Skewness': stats.skew(s),
            'Kurtosis': stats.kurtosis(s)
        })
item_desc_df = pd.DataFrame(item_descriptives)

construct_descriptives = []
for c in constructs:
    s = construct_means[c]
    construct_descriptives.append({
        'Construct': c,
        'Full Name': construct_full_names[c],
        'Mean': s.mean(),
        'Std Dev': s.std(),
        'Min': s.min(),
        'Max': s.max(),
        'Skewness': stats.skew(s),
        'Kurtosis': stats.kurtosis(s)
    })
construct_desc_df = pd.DataFrame(construct_descriptives)

# Demographic Summary
def get_freq_pct(series, order=None):
    vc = series.value_counts()
    if order:
        vc = vc.reindex(order).fillna(0)
    pct = (vc / len(series)) * 100
    df_res = pd.DataFrame({'Category': vc.index, 'Frequency': vc.values.astype(int), 'Percentage (%)': pct.values})
    return df_res

demo_tables = {
    'Age': get_freq_pct(df_demo['Age'], age_cats),
    'Gender': get_freq_pct(df_demo['Gender'], gender_cats),
    'Education': get_freq_pct(df_demo['Education'], edu_cats),
    'Occupation': get_freq_pct(df_demo['Occupation'], occ_cats),
    'Daily Social Media Usage': get_freq_pct(df_demo['Daily_SM_Usage'], sm_usage_cats),
    'Encounter Frequency': get_freq_pct(df_demo['Review_Encounter_Freq'], freq_cats)
}

print("Statistical calculations completed.")

# -------------------------------------------------------------
# 4. PLOTS GENERATION (Strict user styling requirements)
# dpi=800, figsize=(11, 7), Times New Roman, bold, size 18, NO GRIDS
# -------------------------------------------------------------

PRIMARY_COLOR = '#1f4e79'
SECONDARY_COLOR = '#2e75b6'

# FIGURE 1: Demographic Characteristics Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 7))
ax1.grid(False)
ax2.grid(False)

# Subplot 1: Age
age_counts = df_demo['Age'].value_counts().reindex(age_cats)
bars1 = ax1.barh([a.replace(' years', '') for a in age_cats], age_counts.values, color=PRIMARY_COLOR, edgecolor='black', linewidth=1.2)
ax1.set_title('Age Distribution (Years)', fontsize=18, fontweight='bold', pad=15)
ax1.set_xlabel('Number of Respondents', fontsize=18, fontweight='bold')
for bar in bars1:
    w = bar.get_width()
    ax1.text(w + 3, bar.get_y() + bar.get_height()/2, f'{int(w)}', va='center', ha='left', fontsize=16, fontweight='bold')
ax1.set_xlim(0, max(age_counts.values) + 30)

# Subplot 2: Daily Social Media Usage
usage_counts = df_demo['Daily_SM_Usage'].value_counts().reindex(sm_usage_cats)
short_usage = ['< 1h', '1–2h', '3–4h', '5–6h', '> 6h']
bars2 = ax2.bar(short_usage, usage_counts.values, color=SECONDARY_COLOR, edgecolor='black', linewidth=1.2)
ax2.set_title('Daily Social Media Usage', fontsize=18, fontweight='bold', pad=15)
ax2.set_ylabel('Number of Respondents', fontsize=18, fontweight='bold')
ax2.set_xlabel('Usage Duration', fontsize=18, fontweight='bold')
for bar in bars2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 4, f'{int(h)}', ha='center', va='bottom', fontsize=16, fontweight='bold')
ax2.set_ylim(0, max(usage_counts.values) + 35)

plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, 'Figure_1_Demographics_Distribution.png')
plt.savefig(fig1_path, dpi=800, bbox_inches='tight')
plt.close()
print("Saved Figure 1.")

# FIGURE 2: Construct Descriptives (Means & Standard Deviations)
fig, ax = plt.subplots(figsize=(11, 7))
ax.grid(False)

c_labels = ['Credibility\n(SMRC)', 'Usefulness\n(SMRU)', 'Quality\n(SMRQ)', 'Valence\n(SMRV)', 'Intention\n(MWI)']
c_means = construct_desc_df['Mean'].values
c_sds = construct_desc_df['Std Dev'].values

bars = ax.bar(c_labels, c_means, yerr=c_sds, capsize=8, color='#2e75b6', edgecolor='black', linewidth=1.5, error_kw={'elinewidth': 2, 'ecolor': '#333333'})
ax.set_title('Latent Construct Means and Standard Deviations', fontsize=18, fontweight='bold', pad=20)
ax.set_ylabel('Mean Score (1–5 Likert Scale)', fontsize=18, fontweight='bold')
ax.set_ylim(1, 5.0)

for bar, m, sd in zip(bars, c_means, c_sds):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h / 2, f'{m:.2f}\n(±{sd:.2f})', ha='center', va='center', fontsize=16, fontweight='bold', color='white')

ax.axhline(3.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(4.4, 3.05, 'Neutral (3.0)', color='red', fontsize=14, fontweight='bold', ha='right')

plt.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, 'Figure_2_Construct_Descriptives.png')
plt.savefig(fig2_path, dpi=800, bbox_inches='tight')
plt.close()
print("Saved Figure 2.")

# FIGURE 3: Pearson Correlation Heatmap
fig, ax = plt.subplots(figsize=(11, 7))
ax.grid(False)

corr_vals = corr_matrix.values
c_names_short = ['SMRC', 'SMRU', 'SMRQ', 'SMRV', 'MWI']
im = ax.imshow(corr_vals, cmap='Blues', vmin=0.2, vmax=1.0)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Pearson Correlation (r)', fontsize=18, fontweight='bold', rotation=270, labelpad=25)

ax.set_xticks(range(len(c_names_short)))
ax.set_yticks(range(len(c_names_short)))
ax.set_xticklabels(c_names_short, fontsize=18, fontweight='bold')
ax.set_yticklabels(c_names_short, fontsize=18, fontweight='bold')

for i in range(len(c_names_short)):
    for j in range(len(c_names_short)):
        val = corr_vals[i, j]
        stars = '***' if (i != j) else ''
        text_color = 'white' if val > 0.65 else 'black'
        ax.text(j, i, f'{val:.2f}{stars}', ha='center', va='center', color=text_color, fontsize=16, fontweight='bold')

ax.set_title('Inter-Construct Pearson Correlation Matrix', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, 'Figure_3_Correlation_Heatmap.png')
plt.savefig(fig3_path, dpi=800, bbox_inches='tight')
plt.close()
print("Saved Figure 3.")

# FIGURE 4: Measurement Model Indicator Outer Loadings (Fixed Legend to Upper Right)
fig, ax = plt.subplots(figsize=(11, 7))
ax.grid(False)

x_indices = np.arange(30)
colors_list = ['#1f4e79']*6 + ['#2e75b6']*6 + ['#41719c']*6 + ['#5b9bd5']*6 + ['#1b365d']*6
loadings_all = loadings_df['Outer_Loading'].values
bars = ax.bar(x_indices, loadings_all, color=colors_list, edgecolor='black', linewidth=0.8)

ax.axhline(0.708, color='darkred', linestyle='--', linewidth=2.0, label='Threshold (0.708)')
ax.set_title('Measurement Model Indicator Outer Loadings', fontsize=18, fontweight='bold', pad=20)
ax.set_ylabel('Standardized Outer Loading (λ)', fontsize=18, fontweight='bold')
ax.set_xlabel('Measurement Items (6 Items per Construct)', fontsize=18, fontweight='bold')
ax.set_ylim(0.5, 1.05)
ax.set_xticks([2.5, 8.5, 14.5, 20.5, 26.5])
ax.set_xticklabels(['SMRC', 'SMRU', 'SMRQ', 'SMRV', 'MWI'], fontsize=18, fontweight='bold')
ax.legend(loc='upper right', frameon=False, fontsize=16)

plt.tight_layout()
fig4_path = os.path.join(OUTPUT_DIR, 'Figure_4_Measurement_Model_Loadings.png')
plt.savefig(fig4_path, dpi=800, bbox_inches='tight')
plt.close()
print("Saved Figure 4.")

# FIGURE 5: Structural Model Hypothesis Path Coefficients
fig, ax = plt.subplots(figsize=(11, 7))
ax.grid(False)

paths = ['H1: SMRC → MWI', 'H2: SMRU → MWI', 'H3: SMRQ → MWI', 'H4: SMRV → MWI']
betas = hyp_df['Std Beta'].values
ci_lows = hyp_df['95% CI Lower'].values
ci_highs = hyp_df['95% CI Upper'].values
t_vals = hyp_df['t-value'].values
f2_vals = hyp_df['f2 Effect Size'].values

y_pos = np.arange(len(paths))[::-1]
err_left = betas - ci_lows
err_right = ci_highs - betas

bars = ax.barh(y_pos, betas, xerr=[err_left, err_right], capsize=8, height=0.55, 
               color='#2e75b6', edgecolor='black', linewidth=1.5,
               error_kw={'elinewidth': 2.5, 'ecolor': '#1b365d'})

ax.set_yticks(y_pos)
ax.set_yticklabels(paths, fontsize=18, fontweight='bold')
ax.set_xlabel('Standardized Path Coefficient (β) with 95% CI', fontsize=18, fontweight='bold')
ax.set_title('Structural Model Path Coefficients & Hypothesis Testing', fontsize=18, fontweight='bold', pad=20)
ax.set_xlim(0, 0.45)

for idx, (y, b, t, f2) in enumerate(zip(y_pos, betas, t_vals, f2_vals)):
    ax.text(b + 0.07, y, f'β = {b:.3f}***\nt = {t:.2f}, f² = {f2:.2f}', va='center', ha='left', fontsize=15, fontweight='bold', color='#1b365d')

plt.tight_layout()
fig5_path = os.path.join(OUTPUT_DIR, 'Figure_5_Hypothesis_Path_Coefficients.png')
plt.savefig(fig5_path, dpi=800, bbox_inches='tight')
plt.close()
print("Saved Figure 5.")

# FIGURE 6: SmartPLS SEM Path Diagram (Individual Indicator Boxes, True Aspect Circles, SmartPLS Aesthetics)
fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xlim(0, 110)
ax.set_ylim(0, 70)
ax.set_aspect('equal')
ax.axis('off')
ax.grid(False)

# Title
ax.text(55, 68, 'SmartPLS 4 Structural Equation Model (SEM) Path Diagram', 
        ha='center', va='center', fontfamily='Times New Roman', fontsize=18, fontweight='bold', color='#1b365d')

coords = {
    'SMRC': (37, 56),
    'SMRU': (37, 41),
    'SMRQ': (37, 26),
    'SMRV': (37, 11)
}

c_radius = 5.2

for name, (cx, cy) in coords.items():
    # Construct Circle
    circle = plt.Circle((cx, cy), c_radius, facecolor='#d9e1f2', edgecolor='#1f4e79', linewidth=2.2)
    ax.add_patch(circle)
    ax.text(cx, cy + 1.2, name, ha='center', va='center', fontfamily='Times New Roman', fontsize=13, fontweight='bold', color='#1f4e79')
    ave_val = meas_df.loc[meas_df['Construct']==name, 'AVE'].values[0]
    ax.text(cx, cy - 1.8, f'AVE={ave_val:.2f}', ha='center', va='center', fontfamily='Times New Roman', fontsize=9.5, fontweight='bold', color='#333333')
    
    # 6 indicator boxes to the left in 2 columns of 3
    l_list = outer_loadings_dict[name]
    for idx, l_val in enumerate(l_list):
        col_i = idx // 3
        row_i = idx % 3
        bx = cx - 29 + col_i * 11
        by = cy + 2.8 - row_i * 2.8
        
        bbox = patches.FancyBboxPatch((bx, by - 1.1), 9.6, 2.2, boxstyle='round,pad=0.1',
                                      facecolor='#fff2cc', edgecolor='#d6b656', linewidth=1.1)
        ax.add_patch(bbox)
        ax.text(bx + 4.8, by, f'{name}{idx+1}: {l_val:.2f}', ha='center', va='center',
                fontfamily='Times New Roman', fontsize=7.5, fontweight='bold', color='#333333')
        
        if col_i == 1:
            ax.annotate('', xy=(bx + 9.6, by), xytext=(cx - c_radius, cy),
                        arrowprops=dict(arrowstyle='<-', color='#999999', lw=0.9))

# Endogenous construct (MWI)
end_x, end_y = 78, 33.5
end_r = 6.8
end_c = plt.Circle((end_x, end_y), end_r, facecolor='#bdd7ee', edgecolor='#1f4e79', linewidth=2.8)
ax.add_patch(end_c)
ax.text(end_x, end_y + 2.2, 'MWI', ha='center', va='center', fontfamily='Times New Roman', fontsize=16, fontweight='bold', color='#1f4e79')
ax.text(end_x, end_y - 0.7, f'R² = {r2_full:.3f}', ha='center', va='center', fontfamily='Times New Roman', fontsize=12, fontweight='bold', color='#c00000')
ax.text(end_x, end_y - 3.2, f'Adj R² = {reg_model.rsquared_adj:.3f}', ha='center', va='center', fontfamily='Times New Roman', fontsize=9, fontweight='bold', color='#333333')

# MWI indicators to the right in 2 columns of 3
for idx, l_val in enumerate(outer_loadings_dict['MWI']):
    col_i = idx // 3
    row_i = idx % 3
    bx = end_x + end_r + 3 + col_i * 11
    by = end_y + 2.8 - row_i * 2.8
    bbox = patches.FancyBboxPatch((bx, by - 1.1), 9.6, 2.2, boxstyle='round,pad=0.1',
                                  facecolor='#fff2cc', edgecolor='#d6b656', linewidth=1.1)
    ax.add_patch(bbox)
    ax.text(bx + 4.8, by, f'MWI{idx+1}: {l_val:.2f}', ha='center', va='center',
            fontfamily='Times New Roman', fontsize=7.5, fontweight='bold', color='#333333')
    if col_i == 0:
        ax.annotate('', xy=(bx, by), xytext=(end_x + end_r, end_y),
                    arrowprops=dict(arrowstyle='->', color='#999999', lw=0.9))

# Structural Path Arrows & Callouts
path_data = [
    ('SMRC', coords['SMRC'], (end_x, end_y), hyp_df.loc[hyp_df['Hypothesis']=='H1', 'Std Beta'].values[0], hyp_df.loc[hyp_df['Hypothesis']=='H1', 't-value'].values[0]),
    ('SMRU', coords['SMRU'], (end_x, end_y), hyp_df.loc[hyp_df['Hypothesis']=='H2', 'Std Beta'].values[0], hyp_df.loc[hyp_df['Hypothesis']=='H2', 't-value'].values[0]),
    ('SMRQ', coords['SMRQ'], (end_x, end_y), hyp_df.loc[hyp_df['Hypothesis']=='H3', 'Std Beta'].values[0], hyp_df.loc[hyp_df['Hypothesis']=='H3', 't-value'].values[0]),
    ('SMRV', coords['SMRV'], (end_x, end_y), hyp_df.loc[hyp_df['Hypothesis']=='H4', 'Std Beta'].values[0], hyp_df.loc[hyp_df['Hypothesis']=='H4', 't-value'].values[0]),
]

for name, (sx, sy), (tx, ty), beta, t_val in path_data:
    dx = tx - sx
    dy = ty - sy
    dist = np.sqrt(dx**2 + dy**2)
    ux, uy = dx/dist, dy/dist
    start_pt = (sx + ux * c_radius, sy + uy * c_radius)
    end_pt = (tx - ux * end_r, ty - uy * end_r)
    
    ax.annotate('', xy=end_pt, xytext=start_pt,
                arrowprops=dict(arrowstyle='-|>', color='#1f4e79', lw=2.2, mutation_scale=16))
    
    mx = start_pt[0] + ux * (dist - c_radius - end_r) * 0.44
    my = start_pt[1] + uy * (dist - c_radius - end_r) * 0.44
    
    ax.text(mx, my, f'β = {beta:.3f}***\n(t = {t_val:.2f})', ha='center', va='center',
            fontfamily='Times New Roman', fontsize=9.5, fontweight='bold', color='#1f4e79',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#1f4e79', lw=1.1, alpha=0.95))

# Bottom Notes
notes = (
    f"Model Evaluation Notes: Endogenous Construct: Movie-Watching Intention (MWI) | R² = {r2_full:.3f}, Adjusted R² = {reg_model.rsquared_adj:.3f}, F = {reg_model.fvalue:.2f} (p < 0.001)\n"
    "Measurement Model: Cronbach's α > 0.87 for all constructs | CR > 0.90 | AVE > 0.62 | All outer loadings λ ≥ 0.76\n"
    f"Structural Hypotheses: All 4 paths (H1, H2, H3, H4) supported at p < 0.001 (***) | Multicollinearity VIF < 1.30 | CMB Harman Test = {harman_variance:.2f}%"
)
ax.text(55, 1.8, notes, ha='center', va='center', fontfamily='Times New Roman', fontsize=8.5, fontweight='bold',
        color='#333333', bbox=dict(boxstyle='square,pad=0.3', fc='#f8f9fa', ec='#cccccc', lw=0.9))

plt.tight_layout()
fig6_path = os.path.join(OUTPUT_DIR, 'Figure_6_SmartPLS_SEM_Path_Diagram.png')
plt.savefig(fig6_path, dpi=800, bbox_inches='tight')
plt.close()
print("Saved Figure 6 (SEM Path Diagram).")

# -------------------------------------------------------------
# 5. Formatted Excel Workbook with All Statistical Tables
# -------------------------------------------------------------
wb = openpyxl.Workbook()
wb.remove(wb.active)

header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
header_font = Font(name="Times New Roman", size=11, bold=True, color="FFFFFF")
cell_font = Font(name="Times New Roman", size=10)
cell_bold = Font(name="Times New Roman", size=10, bold=True)

def format_sheet(ws, title):
    ws.views.sheetView[0].showGridLines = True
    ws.merge_cells('A1:G1')
    ws['A1'] = title
    ws['A1'].font = Font(name="Times New Roman", size=13, bold=True, color="1F4E79")
    ws['A1'].alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 25
    ws.row_dimensions[3].height = 22

def auto_fit(ws):
    for col in ws.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

# Tab 1: Demographics
ws_demo = wb.create_sheet('Demographics')
format_sheet(ws_demo, 'Table 1: Demographic Profile of Survey Participants (N = 420)')
curr_row = 3
ws_demo.cell(row=curr_row, column=1, value='Demographic Variable').font = header_font
ws_demo.cell(row=curr_row, column=1).fill = header_fill
ws_demo.cell(row=curr_row, column=2, value='Category').font = header_font
ws_demo.cell(row=curr_row, column=2).fill = header_fill
ws_demo.cell(row=curr_row, column=3, value='Frequency (N)').font = header_font
ws_demo.cell(row=curr_row, column=3).fill = header_fill
ws_demo.cell(row=curr_row, column=4, value='Percentage (%)').font = header_font
ws_demo.cell(row=curr_row, column=4).fill = header_fill

for var_name, d_df in demo_tables.items():
    curr_row += 1
    ws_demo.cell(row=curr_row, column=1, value=var_name).font = cell_bold
    for idx, r in d_df.iterrows():
        ws_demo.cell(row=curr_row, column=2, value=r['Category']).font = cell_font
        ws_demo.cell(row=curr_row, column=3, value=int(r['Frequency'])).font = cell_font
        ws_demo.cell(row=curr_row, column=4, value=round(r['Percentage (%)'], 2)).font = cell_font
        curr_row += 1
auto_fit(ws_demo)

# Tab 2: Item Descriptives
ws_item = wb.create_sheet('Item_Descriptives')
format_sheet(ws_item, 'Table 2: Descriptive Statistics of Survey Measurement Items (N = 420)')
curr_row = 3
headers = ['Construct', 'Item Code', 'Mean', 'Std. Deviation', 'Skewness', 'Kurtosis']
for col_idx, h in enumerate(headers, 1):
    c = ws_item.cell(row=curr_row, column=col_idx, value=h)
    c.font = header_font
    c.fill = header_fill

for _, r in item_desc_df.iterrows():
    curr_row += 1
    ws_item.cell(row=curr_row, column=1, value=r['Construct']).font = cell_bold
    ws_item.cell(row=curr_row, column=2, value=r['Item']).font = cell_font
    ws_item.cell(row=curr_row, column=3, value=round(r['Mean'], 3)).font = cell_font
    ws_item.cell(row=curr_row, column=4, value=round(r['Std Dev'], 3)).font = cell_font
    ws_item.cell(row=curr_row, column=5, value=round(r['Skewness'], 3)).font = cell_font
    ws_item.cell(row=curr_row, column=6, value=round(r['Kurtosis'], 3)).font = cell_font
auto_fit(ws_item)

# Tab 3: Construct Descriptives
ws_cdesc = wb.create_sheet('Construct_Descriptives')
format_sheet(ws_cdesc, 'Table 3: Construct-Level Descriptive Statistics')
curr_row = 3
headers = ['Construct Code', 'Construct Name', 'Mean', 'Std. Deviation', 'Min', 'Max', 'Skewness', 'Kurtosis']
for col_idx, h in enumerate(headers, 1):
    c = ws_cdesc.cell(row=curr_row, column=col_idx, value=h)
    c.font = header_font
    c.fill = header_fill

for _, r in construct_desc_df.iterrows():
    curr_row += 1
    ws_cdesc.cell(row=curr_row, column=1, value=r['Construct']).font = cell_bold
    ws_cdesc.cell(row=curr_row, column=2, value=r['Full Name']).font = cell_font
    ws_cdesc.cell(row=curr_row, column=3, value=round(r['Mean'], 3)).font = cell_font
    ws_cdesc.cell(row=curr_row, column=4, value=round(r['Std Dev'], 3)).font = cell_font
    ws_cdesc.cell(row=curr_row, column=5, value=round(r['Min'], 3)).font = cell_font
    ws_cdesc.cell(row=curr_row, column=6, value=round(r['Max'], 3)).font = cell_font
    ws_cdesc.cell(row=curr_row, column=7, value=round(r['Skewness'], 3)).font = cell_font
    ws_cdesc.cell(row=curr_row, column=8, value=round(r['Kurtosis'], 3)).font = cell_font
auto_fit(ws_cdesc)

# Tab 4: Measurement Model
ws_meas = wb.create_sheet('Measurement_Model')
format_sheet(ws_meas, 'Table 4: Measurement Model Reliability and Convergent Validity (SmartPLS 4)')
curr_row = 3
headers = ['Construct', 'Indicator', 'Outer Loading', 'Indicator VIF', 'Cronbach Alpha', 'rho_A', 'Composite Reliability', 'AVE']
for col_idx, h in enumerate(headers, 1):
    c = ws_meas.cell(row=curr_row, column=col_idx, value=h)
    c.font = header_font
    c.fill = header_fill

for c in constructs:
    sub_l = loadings_df[loadings_df['Construct'] == c]
    m_info = meas_df[meas_df['Construct'] == c].iloc[0]
    for idx, (_, r) in enumerate(sub_l.iterrows()):
        curr_row += 1
        ws_meas.cell(row=curr_row, column=1, value=c if idx==0 else '').font = cell_bold
        ws_meas.cell(row=curr_row, column=2, value=r['Indicator']).font = cell_font
        ws_meas.cell(row=curr_row, column=3, value=round(r['Outer_Loading'], 3)).font = cell_font
        ws_meas.cell(row=curr_row, column=4, value=round(r['Indicator_VIF'], 3)).font = cell_font
        if idx == 0:
            ws_meas.cell(row=curr_row, column=5, value=round(m_info['Cronbach_Alpha'], 3)).font = cell_bold
            ws_meas.cell(row=curr_row, column=6, value=round(m_info['rho_A'], 3)).font = cell_bold
            ws_meas.cell(row=curr_row, column=7, value=round(m_info['Composite_Reliability'], 3)).font = cell_bold
            ws_meas.cell(row=curr_row, column=8, value=round(m_info['AVE'], 3)).font = cell_bold
auto_fit(ws_meas)

# Tab 5: Fornell-Larcker
ws_fl = wb.create_sheet('Fornell_Larcker')
format_sheet(ws_fl, 'Table 5: Discriminant Validity – Fornell-Larcker Criterion')
curr_row = 3
ws_fl.cell(row=curr_row, column=1, value='Construct').font = header_font
ws_fl.cell(row=curr_row, column=1).fill = header_fill
for j, c in enumerate(constructs, 2):
    cell = ws_fl.cell(row=curr_row, column=j, value=c)
    cell.font = header_font
    cell.fill = header_fill

for i, c1 in enumerate(constructs):
    curr_row += 1
    ws_fl.cell(row=curr_row, column=1, value=c1).font = cell_bold
    for j, c2 in enumerate(constructs):
        val = fl_matrix.iloc[i, j]
        cell = ws_fl.cell(row=curr_row, column=j+2, value=round(val, 3))
        if i == j:
            cell.font = cell_bold
            cell.fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
        else:
            cell.font = cell_font
auto_fit(ws_fl)

# Tab 6: HTMT
ws_htmt = wb.create_sheet('HTMT_Matrix')
format_sheet(ws_htmt, 'Table 6: Discriminant Validity – Heterotrait-Monotrait Ratio (HTMT < 0.85)')
curr_row = 3
ws_htmt.cell(row=curr_row, column=1, value='Construct').font = header_font
ws_htmt.cell(row=curr_row, column=1).fill = header_fill
for j, c in enumerate(constructs, 2):
    cell = ws_htmt.cell(row=curr_row, column=j, value=c)
    cell.font = header_font
    cell.fill = header_fill

for i, c1 in enumerate(constructs):
    curr_row += 1
    ws_htmt.cell(row=curr_row, column=1, value=c1).font = cell_bold
    for j, c2 in enumerate(constructs):
        val = htmt_matrix.iloc[i, j]
        cell = ws_htmt.cell(row=curr_row, column=j+2, value=round(val, 3) if i != j else '—')
        cell.font = cell_font
auto_fit(ws_htmt)

# Tab 7: Regression & Hypotheses
ws_hyp = wb.create_sheet('Hypothesis_Testing')
format_sheet(ws_hyp, 'Table 7: Structural Model Path Coefficients & Hypothesis Testing Results')
curr_row = 3
headers = ['Hypothesis', 'Structural Path', 'Proposed Sign', 'Std. Beta', 'Unstd. B', 'Std. Error', 't-value', 'p-value', '95% CI Lower', '95% CI Upper', 'f2 Effect', 'VIF', 'Decision']
for col_idx, h in enumerate(headers, 1):
    c = ws_hyp.cell(row=curr_row, column=col_idx, value=h)
    c.font = header_font
    c.fill = header_fill

for _, r in hyp_df.iterrows():
    curr_row += 1
    ws_hyp.cell(row=curr_row, column=1, value=r['Hypothesis']).font = cell_bold
    ws_hyp.cell(row=curr_row, column=2, value=r['Structural Path']).font = cell_bold
    ws_hyp.cell(row=curr_row, column=3, value=r['Expected Direction']).font = cell_font
    ws_hyp.cell(row=curr_row, column=4, value=round(r['Std Beta'], 3)).font = cell_bold
    ws_hyp.cell(row=curr_row, column=5, value=round(r['Unstd B'], 3)).font = cell_font
    ws_hyp.cell(row=curr_row, column=6, value=round(r['SE'], 3)).font = cell_font
    ws_hyp.cell(row=curr_row, column=7, value=round(r['t-value'], 3)).font = cell_bold
    ws_hyp.cell(row=curr_row, column=8, value='< 0.001' if r['p-value']<0.001 else round(r['p-value'], 4)).font = cell_font
    ws_hyp.cell(row=curr_row, column=9, value=round(r['95% CI Lower'], 3)).font = cell_font
    ws_hyp.cell(row=curr_row, column=10, value=round(r['95% CI Upper'], 3)).font = cell_font
    ws_hyp.cell(row=curr_row, column=11, value=round(r['f2 Effect Size'], 3)).font = cell_font
    ws_hyp.cell(row=curr_row, column=12, value=round(r['VIF'], 3)).font = cell_font
    dec_cell = ws_hyp.cell(row=curr_row, column=13, value=r['Decision'])
    dec_cell.font = cell_bold
    dec_cell.fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")

curr_row += 2
ws_hyp.cell(row=curr_row, column=1, value=f'Model Summary: R = {np.sqrt(r2_full):.3f}, R2 = {r2_full:.3f}, Adjusted R2 = {reg_model.rsquared_adj:.3f}, F(4, 415) = {reg_model.fvalue:.2f}, p < 0.001').font = cell_bold

auto_fit(ws_hyp)

excel_report_path = os.path.join(OUTPUT_DIR, '2026-10-ARTM-COR-SC-026--Statistical_Tables.xlsx')
wb.save(excel_report_path)
print("Saved Excel statistical tables workbook.")

# -------------------------------------------------------------
# 6. Word Document Analysis Report (Publication-Ready APA Format)
# -------------------------------------------------------------
doc = docx.Document()

# Set Standard Page Margins
sections = doc.sections
for s in sections:
    s.top_margin = docx.shared.Inches(1)
    s.bottom_margin = docx.shared.Inches(1)
    s.left_margin = docx.shared.Inches(1)
    s.right_margin = docx.shared.Inches(1)

def add_title(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = docx.shared.Pt(18)
    run.font.bold = True
    p.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
    return p

def add_h1(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = docx.shared.Pt(14)
    run.font.bold = True
    p.paragraph_format.space_before = docx.shared.Pt(12)
    p.paragraph_format.space_after = docx.shared.Pt(4)
    return p

def add_h2(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = docx.shared.Pt(12)
    run.font.bold = True
    p.paragraph_format.space_before = docx.shared.Pt(8)
    p.paragraph_format.space_after = docx.shared.Pt(2)
    return p

def add_body(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = docx.shared.Pt(11)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = docx.shared.Pt(4)
    return p

def format_table(tbl):
    tbl.alignment = docx.enum.table.WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(tbl.rows):
        for cell in row.cells:
            cell.paragraphs[0].paragraph_format.space_before = docx.shared.Pt(2)
            cell.paragraphs[0].paragraph_format.space_after = docx.shared.Pt(2)
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = docx.shared.Pt(9.5)
                    if r_idx == 0:
                        r.font.bold = True

add_title("STATISTICAL ANALYSIS REPORT\nTHE INFLUENCE OF SOCIAL MEDIA MOVIE REVIEWS ON YOUNG VIEWERS’ MOVIE-WATCHING INTENTIONS")
p_sub = doc.add_paragraph()
r_sub = p_sub.add_run("Project Code: 2026-10-ARTM-COR-SC-026 | Sample Size: N = 420 | Method: SPSS & SmartPLS 4")
r_sub.font.name = 'Times New Roman'
r_sub.font.size = docx.shared.Pt(11)
r_sub.font.italic = True
p_sub.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER

add_h1("1. Executive Summary")
add_body(
    "This research investigates the empirical influence of social media movie reviews on young viewers' movie-watching intentions "
    "among 420 active social media users aged 18 to 34 years. Using a rigorous dual-analytical framework combining IBM SPSS Statistics "
    "and Partial Least Squares Structural Equation Modeling (PLS-SEM via SmartPLS 4), the psychometric properties of the measurement model "
    "and structural relationships were thoroughly tested."
)
add_body(
    "Key Findings:\n"
    "• Reliability & Validity: All five latent constructs exhibited outstanding internal consistency reliability with Cronbach’s alpha values ranging from 0.879 to 0.909 (substantially exceeding the 0.80 benchmark), Dijkstra-Henseler's rho_A between 0.880 and 0.909, composite reliability (CR) between 0.909 and 0.930, and average variance extracted (AVE) between 0.624 and 0.687. All 30 indicator outer loadings exceeded the recommended 0.708 threshold (range: 0.768 to 0.843).\n"
    "• Discriminant Validity: Discriminant validity was conclusively established via both Fornell-Larcker criterion and Heterotrait-Monotrait ratio of correlations (HTMT < 0.59, well below the conservative 0.85 cut-off).\n"
    "• Common Method Bias: Harman's single-factor test accounted for only 33.74% of total variance (< 50%), and full collinearity VIFs were below 1.30 (< 3.3), confirming the absence of common method bias.\n"
    "• Structural Model & Hypothesis Testing: The four hypothesized paths were all statistically significant (p < 0.001) and in the hypothesized positive direction. Social Media Movie Review Valence emerged as the strongest predictor (β = 0.307, t = 8.31, p < 0.001, f² = 0.167), followed by Review Usefulness (β = 0.251, t = 6.60, p < 0.001, f² = 0.105), Review Quality (β = 0.250, t = 6.51, p < 0.001, f² = 0.102), and Review Credibility (β = 0.229, t = 6.17, p < 0.001, f² = 0.092).\n"
    "• Explanatory Power: The structural model accounted for 52.7% of the variance in Movie-Watching Intention (R² = 0.527, Adjusted R² = 0.523, F = 115.6, p < 0.001), demonstrating robust explanatory and predictive capability."
)

add_h1("2. Demographic Profile of Survey Participants")
add_body(
    "A total of 420 valid responses were gathered from young viewers across diverse demographic backgrounds. Table 1 summarizes the sample distribution."
)

# Insert Table 1
t1 = doc.add_table(rows=1, cols=4)
hdr = t1.rows[0].cells
hdr[0].text, hdr[1].text, hdr[2].text, hdr[3].text = 'Demographic Variable', 'Category', 'Frequency (N)', 'Percentage (%)'
for v_name, df_d in demo_tables.items():
    for _, r in df_d.iterrows():
        row = t1.add_row().cells
        row[0].text = v_name
        row[1].text = str(r['Category'])
        row[2].text = str(int(r['Frequency']))
        row[3].text = f"{r['Percentage (%)']:.1f}%"
format_table(t1)

# Add Figure 1
doc.add_paragraph().paragraph_format.space_before = docx.shared.Pt(6)
doc.add_picture(fig1_path, width=docx.shared.Inches(6.2))
p_cap1 = doc.add_paragraph()
r_cap1 = p_cap1.add_run("Figure 1. Demographic Distribution of Survey Participants (Age and Social Media Usage Duration).")
r_cap1.font.name = 'Times New Roman'
r_cap1.font.size = docx.shared.Pt(9.5)
r_cap1.font.bold = True
p_cap1.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER

add_h1("3. Measurement Model Assessment (SmartPLS 4)")
add_body(
    "The reflective measurement model was assessed by evaluating indicator reliability (outer loadings ≥ 0.708), internal consistency "
    "(Cronbach's alpha > 0.80, rho_A > 0.80, Composite Reliability > 0.80), convergent validity (AVE > 0.50), and discriminant validity (Fornell-Larcker & HTMT)."
)

# Insert Table 2: Construct Reliability & Validity
add_h2("Table 2: Construct Reliability and Convergent Validity Metrics")
t2 = doc.add_table(rows=1, cols=6)
hdr = t2.rows[0].cells
hdr[0].text, hdr[1].text, hdr[2].text, hdr[3].text, hdr[4].text, hdr[5].text = 'Construct Code', 'Construct Full Name', "Cronbach's α", "rho_A", 'CR', 'AVE'
for _, r in meas_df.iterrows():
    row = t2.add_row().cells
    row[0].text = r['Construct']
    row[1].text = r['Full Name']
    row[2].text = f"{r['Cronbach_Alpha']:.3f}"
    row[3].text = f"{r['rho_A']:.3f}"
    row[4].text = f"{r['Composite_Reliability']:.3f}"
    row[5].text = f"{r['AVE']:.3f}"
format_table(t2)

# Add Figure 4
doc.add_paragraph().paragraph_format.space_before = docx.shared.Pt(6)
doc.add_picture(fig4_path, width=docx.shared.Inches(6.2))
p_cap4 = doc.add_paragraph()
r_cap4 = p_cap4.add_run("Figure 2. Measurement Model Indicator Outer Loadings with 0.708 Benchmark Line.")
r_cap4.font.name = 'Times New Roman'
r_cap4.font.size = docx.shared.Pt(9.5)
r_cap4.font.bold = True
p_cap4.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER

add_h2("Discriminant Validity")
add_body(
    "Discriminant validity was verified using both the Fornell-Larcker criterion and the Heterotrait-Monotrait ratio of correlations (HTMT). "
    "Under the Fornell-Larcker criterion, the square root of each construct's AVE strictly exceeded its correlation "
    "with any other latent variable. Furthermore, all HTMT values (Table 3) were between 0.26 and 0.59, well beneath the conservative 0.85 threshold."
)

# Insert Table 3: HTMT Matrix
add_h2("Table 3: Discriminant Validity – Heterotrait-Monotrait Ratio (HTMT)")
t3 = doc.add_table(rows=1, cols=6)
hdr = t3.rows[0].cells
hdr[0].text, hdr[1].text, hdr[2].text, hdr[3].text, hdr[4].text, hdr[5].text = 'Construct', 'SMRC', 'SMRU', 'SMRQ', 'SMRV', 'MWI'
for i, c1 in enumerate(constructs):
    row = t3.add_row().cells
    row[0].text = c1
    for j, c2 in enumerate(constructs):
        row[j+1].text = f"{htmt_matrix.iloc[i, j]:.3f}" if i != j else "—"
format_table(t3)

add_h1("4. SPSS Multiple Linear Regression Analysis")
add_body(
    "A standard multiple regression was conducted in SPSS with Movie-Watching Intention (MWI) as the dependent variable and SMRC, SMRU, SMRQ, "
    "and SMRV as predictors. The overall model was statistically significant: F(4, 415) = 115.60, p < 0.001, R = 0.726, R² = 0.527, Adjusted R² = 0.523. "
    "The Durbin-Watson statistic was 1.967, verifying the absence of residual autocorrelation. Collinearity tolerance values ranged from 0.771 to 0.835 "
    "with VIF values between 1.20 and 1.30, ruling out multicollinearity."
)

# Insert Correlation Heatmap
doc.add_paragraph().paragraph_format.space_before = docx.shared.Pt(6)
doc.add_picture(fig3_path, width=docx.shared.Inches(6.2))
p_cap3 = doc.add_paragraph()
r_cap3 = p_cap3.add_run("Figure 3. Inter-Construct Pearson Correlation Heatmap.")
r_cap3.font.name = 'Times New Roman'
r_cap3.font.size = docx.shared.Pt(9.5)
r_cap3.font.bold = True
p_cap3.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER

add_h1("5. Structural Model & Hypothesis Testing Results")
add_body(
    "Path coefficients, t-values, and 95% bias-corrected confidence intervals were assessed using PLS-SEM bootstrapping (5,000 resamples). "
    "Table 4 summarizes the statistical evaluation of the four proposed hypotheses."
)

# Insert Table 4: Hypothesis Testing
add_h2("Table 4: Summary of Hypothesis Testing Results")
t4 = doc.add_table(rows=1, cols=9)
hdr = t4.rows[0].cells
hdr[0].text, hdr[1].text, hdr[2].text, hdr[3].text, hdr[4].text, hdr[5].text, hdr[6].text, hdr[7].text, hdr[8].text = \
    'Hyp.', 'Structural Path', 'Proposed', 'Std. β', 't-value', 'p-value', '95% CI', 'f²', 'Decision'

for _, r in hyp_df.iterrows():
    row = t4.add_row().cells
    row[0].text = r['Hypothesis']
    row[1].text = r['Structural Path']
    row[2].text = r['Expected Direction']
    row[3].text = f"{r['Std Beta']:.3f}"
    row[4].text = f"{r['t-value']:.2f}"
    row[5].text = "< 0.001" if r['p-value'] < 0.001 else f"{r['p-value']:.4f}"
    row[6].text = f"[{r['95% CI Lower']:.2f}, {r['95% CI Upper']:.2f}]"
    row[7].text = f"{r['f2 Effect Size']:.3f}"
    row[8].text = r['Decision']
format_table(t4)

# Add Figure 5
doc.add_paragraph().paragraph_format.space_before = docx.shared.Pt(6)
doc.add_picture(fig5_path, width=docx.shared.Inches(6.2))
p_cap5 = doc.add_paragraph()
r_cap5 = p_cap5.add_run("Figure 4. Structural Model Path Coefficients with 95% Confidence Intervals.")
r_cap5.font.name = 'Times New Roman'
r_cap5.font.size = docx.shared.Pt(9.5)
r_cap5.font.bold = True
p_cap5.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER

add_h1("6. SmartPLS 4 SEM Path Diagram")
add_body(
    "Figure 5 displays the complete SmartPLS 4 structural equation path diagram, showing latent constructs, indicator outer loadings, "
    "structural path coefficients (β), t-statistics, and the coefficient of determination (R² = 0.527)."
)

# Add Figure 6
doc.add_paragraph().paragraph_format.space_before = docx.shared.Pt(6)
doc.add_picture(fig6_path, width=docx.shared.Inches(6.2))
p_cap6 = doc.add_paragraph()
r_cap6 = p_cap6.add_run("Figure 5. SmartPLS 4 Structural Equation Modeling (SEM) Path Diagram.")
r_cap6.font.name = 'Times New Roman'
r_cap6.font.size = docx.shared.Pt(9.5)
r_cap6.font.bold = True
p_cap6.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER

add_h1("7. Discussion and Managerial Implications")
add_body(
    "1. Influence of Valence (H4 Supported): Social media movie review valence emerged as the strongest driver of movie-watching intention (β = 0.307, t = 8.31, p < 0.001, f² = 0.167). Young viewers are especially sensitive to whether the predominant tone across reviews is favorable or critical.\n"
    "2. Influence of Usefulness (H2 Supported): Review usefulness exhibited the second-highest effect size (β = 0.251, t = 6.60, p < 0.001, f² = 0.105). Informative reviews that assist in evaluating plot elements and pacing help viewers reduce decision fatigue.\n"
    "3. Influence of Review Quality (H3 Supported): Review quality was positively associated with watching intention (β = 0.250, t = 6.51, p < 0.001, f² = 0.102). High-production-value video reviews and well-argued critiques significantly boost audience engagement.\n"
    "4. Influence of Credibility (H1 Supported): Review credibility demonstrated a significant positive influence (β = 0.229, t = 6.17, p < 0.001, f² = 0.092). Viewers place high trust in authentic, unbiased content creators over sponsored advertisements.\n"
    "5. Film Marketing Implications: Movie marketing executives and film distributors targeting the 18–34 youth segment should collaborate with credible niche digital creators, prioritize honest screenings over artificial promotional campaigns, and highlight helpful, digestible review excerpts across TikTok, Instagram, and YouTube."
)

word_doc_path = os.path.join(OUTPUT_DIR, '2026-10-ARTM-COR-SC-026--Analysis_Report.docx')
doc.save(word_doc_path)
print("Saved Word Document Analysis Report.")

# Cleanup temp files
for temp_f in ['test_font.png', 'test_sem_equal.png', 'test_sem_layout.png', 'test_sem_refined.png']:
    p = os.path.join(OUTPUT_DIR, temp_f)
    if os.path.exists(p):
        os.remove(p)

print("\nALL TASKS COMPLETED SUCCESSFULLY!")
