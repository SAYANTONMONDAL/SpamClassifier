# ── IMPORTS ──────────────────────────────────────────────
import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score, roc_curve)

# ── STEP 1: LOAD DATA ────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
print(f"Dataset loaded: {len(df)} messages")
print(df['label'].value_counts())

# ── STEP 2: PREPROCESS TEXT ──────────────────────────────
STOPWORDS = {'i','me','my','we','our','you','your','he','him','his','she',
             'her','it','its','they','them','their','what','this','that',
             'am','is','are','was','were','be','been','have','has','had',
             'do','does','did','a','an','the','and','but','if','or','as',
             'of','at','by','for','with','to','from','in','on','so','not',
             'no','just','will','can','now','its','also'}

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', 'url', text)
    text = re.sub(r'\d+', 'num', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return ' '.join(words)

print("\nPreprocessing text...")
df['clean'] = df['message'].apply(preprocess)
df['label_num'] = (df['label'] == 'spam').astype(int)

# ── STEP 3: FEATURE EXTRACTION ───────────────────────────
print("Extracting TF-IDF features...")
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
X = tfidf.fit_transform(df['clean']).toarray()
y = df['label_num'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── STEP 4: TRAIN ALL MODELS ─────────────────────────────
models = {
    'Naive Bayes':         MultinomialNB(alpha=0.5),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM':                 SVC(kernel='linear', probability=True, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost':            AdaBoostClassifier(n_estimators=100, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=5),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n" + "="*65)
print("TRAINING ALL MODELS...")
print("="*65)

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')

    results[name] = {
        'model':    model,
        'y_pred':   y_pred,
        'y_prob':   y_prob,
        'Accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
        'Precision':round(precision_score(y_test, y_pred) * 100, 2),
        'Recall':   round(recall_score(y_test, y_pred) * 100, 2),
        'F1':       round(f1_score(y_test, y_pred) * 100, 2),
        'ROC-AUC':  round(roc_auc_score(y_test, y_prob) * 100, 2),
        'CV Mean':  round(cv_scores.mean() * 100, 2),
        'CV Std':   round(cv_scores.std() * 100, 2),
        'cm':       confusion_matrix(y_test, y_pred)
    }
    r = results[name]
    print(f"  Acc={r['Accuracy']}% | F1={r['F1']}% | ROC={r['ROC-AUC']}% | CV={r['CV Mean']}%±{r['CV Std']}%")

# ── STEP 5: SUMMARY TABLE ────────────────────────────────
summary = pd.DataFrame({
    name: {k: results[name][k] for k in ['Accuracy','Precision','Recall','F1','ROC-AUC','CV Mean','CV Std']}
    for name in results
}).T.sort_values('F1', ascending=False)

print("\n\nFINAL RESULTS:")
print(summary.to_string())

best_name = summary.index[0]
print(f"\n🏆 BEST MODEL: {best_name} with F1={summary.loc[best_name,'F1']}%")

# ── STEP 6: SAVE CHARTS ──────────────────────────────────
print("\nGenerating charts...")
DARK = '#0F172A'; CARD = '#1E293B'; TEXT = '#F1F5F9'; GRID = '#334155'
PALETTE = ['#4F46E5','#22C55E','#EF4444','#F59E0B','#06B6D4','#A855F7','#F97316','#64748B']
plt.rcParams.update({'figure.facecolor':DARK,'axes.facecolor':CARD,
                     'axes.edgecolor':GRID,'axes.labelcolor':TEXT,
                     'xtick.color':TEXT,'ytick.color':TEXT,'text.color':TEXT,
                     'grid.color':GRID,'grid.linestyle':'--','grid.alpha':0.4})

names_sorted = list(summary.index)
colors = PALETTE[:len(names_sorted)]

# Chart 1 — Metrics Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('Algorithm Comparison — Email Spam Classifier', fontsize=15, fontweight='bold')
for ax, metric in zip(axes.flat, ['Accuracy','Precision','Recall','F1']):
    vals = [results[n][metric] for n in names_sorted]
    bars = ax.bar(range(len(names_sorted)), vals, color=colors, edgecolor='white', width=0.6)
    ax.set_title(metric, fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(names_sorted)))
    ax.set_xticklabels(names_sorted, rotation=30, ha='right', fontsize=9)
    ax.set_ylim(max(0, min(vals)-5), 103)
    ax.grid(axis='y')
    bi = vals.index(max(vals))
    bars[bi].set_edgecolor('#F59E0B'); bars[bi].set_linewidth(3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig('chart1_metrics.png', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()

# Chart 2 — ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot([0,1],[0,1],'--', color=GRID, linewidth=1.5, label='Random (AUC=0.50)')
for name, color in zip(names_sorted, colors):
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_prob'])
    ax.plot(fpr, tpr, color=color, linewidth=2.5,
            label=f"{name} (AUC={results[name]['ROC-AUC']}%)")
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9, framealpha=0.3)
ax.grid(True); ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
plt.tight_layout()
plt.savefig('chart2_roc.png', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()

# Chart 3 — Confusion Matrix of Best Model
fig, ax = plt.subplots(figsize=(6, 5))
cm = results[best_name]['cm']
im = ax.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                fontsize=20, fontweight='bold',
                color='white' if cm[i,j] > cm.max()/2 else TEXT)
ax.set_xticks([0,1]); ax.set_xticklabels(['Ham','Spam'])
ax.set_yticks([0,1]); ax.set_yticklabels(['Ham','Spam'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix — {best_name}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('chart3_confusion.png', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()

# Chart 4 — CV Comparison
fig, ax = plt.subplots(figsize=(10, 6))
cv_means = [results[n]['CV Mean'] for n in names_sorted]
cv_stds  = [results[n]['CV Std'] for n in names_sorted]
ax.barh(names_sorted[::-1], cv_means[::-1], xerr=cv_stds[::-1],
        color=colors[::-1], edgecolor='white',
        error_kw=dict(ecolor='#F59E0B', elinewidth=2, capsize=5))
ax.set_title('Cross-Validation F1 Score (5-Fold)', fontsize=13, fontweight='bold')
ax.set_xlabel('F1 Score (%)')
ax.grid(axis='x')
for i,(m,s) in enumerate(zip(cv_means[::-1], cv_stds[::-1])):
    ax.text(m+s+0.2, i, f'{m:.1f}%', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('chart4_cv.png', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()

print("Charts saved: chart1_metrics.png, chart2_roc.png, chart3_confusion.png, chart4_cv.png")

# ── STEP 7: SAVE BEST MODEL ──────────────────────────────
joblib.dump(results[best_name]['model'], 'best_model.pkl')
joblib.dump(tfidf, 'tfidf.pkl')
summary.to_csv('results.csv')
print(f"\nModel saved! Best: {best_name}")
print("\nDONE! Now run: streamlit run app.py")