import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# ===================== 1.读取数据 =====================
df = pd.read_csv("netflix_titles.csv")
print("原始数据形状：", df.shape)
print(df.head())

# ===================== 2.数据清洗 =====================
# 删除关键列缺失样本
df = df.dropna(subset=["type", "release_year", "duration", "listed_in"])

# 提取时长数字
def extract_duration(x):
    if "min" in str(x):
        return int(x.replace(" min", ""))
    else:
        return np.nan

df["duration_num"] = df["duration"].apply(extract_duration)
df = df.dropna(subset=["duration_num"])

# 取第一个题材
df["main_genre"] = df["listed_in"].str.split(",").str[0].str.strip()

# 构造特征和目标变量
# 特征：节目类型、上映年份、时长、主题材
X = df[["type", "release_year", "duration_num", "main_genre"]]
y = df["release_year"]  # 预测上映年份；如果有imdb评分替换成 y = df["imdb_score"]

# ===================== 3.划分训练集、测试集 8:2 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===================== 4.预处理流水线 =====================
cat_cols = ["type", "main_genre"]
num_cols = ["release_year", "duration_num"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), cat_cols)
    ]
)

# ===================== 5.ElasticNet + 网格搜索调参 =====================
pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("elasticnet", ElasticNet(max_iter=10000))
])

param_grid = {
    "elasticnet__alpha": [0.01, 0.1, 0.5, 1, 5],
    "elasticnet__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
grid.fit(X_train, y_train)

print("\n========= 最优参数 =========")
print(grid.best_params_)
print(f"交叉验证最优MSE: {-grid.best_score_:.4f}")

# ===================== 6.模型评估 =====================
y_pred = grid.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n========= 测试集结果 =========")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# ===================== 7.输出特征系数 =====================
best_model = grid.best_estimator_
feature_names = num_cols + list(
    best_model.named_steps["preprocess"]
    .transformers_[1][1]
    .get_feature_names_out(cat_cols)
)
coef = best_model.named_steps["elasticnet"].coef_

coef_df = pd.DataFrame({"feature": feature_names, "coef": coef})
coef_df = coef_df.sort_values("coef", ascending=False)
print("\n========= 特征系数 =========")
print(coef_df)

# ===================== 绘图 =====================
plt.figure(figsize=(10, 6))
sns.barplot(data=coef_df.head(12), x="coef", y="feature")
plt.title("ElasticNet 特征系数 Top12")
plt.tight_layout()
plt.show()

# 真实值vs预测值
plt.figure(figsize=(5, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("真实值")
plt.ylabel("预测值")
plt.title("真实值 vs 预测值")
plt.tight_layout()
plt.show()
