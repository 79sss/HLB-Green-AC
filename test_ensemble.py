import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE  # 用于平衡数据
from sklearn.cross_decomposition import PLSRegression

# 定义计算 VIP 值的函数
def calculate_vip(pls_model, X):
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        vips[i] = np.sqrt(p * (s.T @ weight).item() / total_s)
    return vips

# 读取数据并处理缺失值
df = pd.read_excel('data5.xlsx')
df.fillna(0, inplace=True)

# 提取特征和标签
X = df.iloc[:, 1:]
y = np.where(df.iloc[:, 0] == 'sick', 1, 0)

# 确保标签一致
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 处理类别不均衡（如果有不均衡问题）
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 划分数据集
xtrain, xtest, ytrain, ytest = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)

# 特征标准化（对于 XGBoost 和 RandomForest，标准化有时能提高效果）
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

# XGBoost 参数调优
xgb = XGBClassifier(random_state=42, eval_metric='logloss')  # 移除 use_label_encoder 参数
param_grid_xgb = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}
grid_xgb = GridSearchCV(xgb, param_grid_xgb, scoring='roc_auc', cv=StratifiedKFold(n_splits=5), n_jobs=-1)
grid_xgb.fit(xtrain, ytrain)
best_xgb = grid_xgb.best_estimator_

# RandomForest 参数调优
rfc = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}
grid_rf = GridSearchCV(rfc, param_grid_rf, scoring='roc_auc', cv=StratifiedKFold(n_splits=5), n_jobs=-1)
grid_rf.fit(xtrain, ytrain)
best_rfc = grid_rf.best_estimator_

# 计算基分类器在验证集上的 AUC 得分
xgb_pred_proba = best_xgb.predict_proba(xtest)[:, 1]
xgb_auc = roc_auc_score(ytest, xgb_pred_proba)

rfc_pred_proba = best_rfc.predict_proba(xtest)[:, 1]
rfc_auc = roc_auc_score(ytest, rfc_pred_proba)

# 根据 AUC 得分计算权重
total_auc = xgb_auc + rfc_auc
weights = [xgb_auc / total_auc, rfc_auc / total_auc]

# 输出各个基分类器的权重
print(f"XGBoost 权重: {weights[0]:.4f}")
print(f"RandomForest 权重: {weights[1]:.4f}")

# 集成模型
ensemble_model = VotingClassifier(estimators=[('xgb', best_xgb), ('rf', best_rfc)], voting='soft', weights=weights)
ensemble_model.fit(xtrain, ytrain)

# 模型预测与评估
y_pred_proba = ensemble_model.predict_proba(xtest)[:, 1]
y_pred = ensemble_model.predict(xtest)

print("Ensemble Test AUC:", roc_auc_score(ytest, y_pred_proba))
print("Ensemble Accuracy:", accuracy_score(ytest, y_pred))

# 绘制 ROC 曲线
fpr, tpr, _ = roc_curve(ytest, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='AUC = %0.4f' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 绘制混淆矩阵
cm = confusion_matrix(ytest, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 绘制特征重要性（XGBoost）
xgb_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': best_xgb.feature_importances_})
xgb_importance_df = xgb_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=xgb_importance_df)
plt.title("XGBoost Feature Importances")
plt.show()

# 绘制特征重要性（RandomForest）
rf_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': best_rfc.feature_importances_})
rf_importance_df = rf_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=rf_importance_df)
plt.title("RandomForest Feature Importances")
plt.show()

# 绘制 Precision - Recall 曲线
precision, recall, _ = precision_recall_curve(ytest, y_pred_proba)
pr_auc = auc(recall, precision)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='PR AUC = %0.4f' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision - Recall Curve')
plt.legend(loc="lower left")
plt.show()

# 计算集成模型的特征重要性（使用根据 AUC 计算的权重）
ensemble_importance = weights[0] * best_xgb.feature_importances_ + weights[1] * best_rfc.feature_importances_
ensemble_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': ensemble_importance})
ensemble_importance_df = ensemble_importance_df.sort_values(by='Importance', ascending=False)

# 绘制集成模型的特征重要性图
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=ensemble_importance_df)
plt.title("Ensemble Model Feature Importances")
plt.show()

# 使用 PLS 回归模型计算 VIP 值
pls = PLSRegression(n_components=2)  # 可以根据实际情况调整成分数量
pls.fit(xtrain, ytrain)
vip_values = calculate_vip(pls, xtrain)

# 筛选 VIP 值大于 1 的特征
vip_df = pd.DataFrame({'Feature': X.columns, 'VIP': vip_values})
vip_df = vip_df[vip_df['VIP'] > 1]

# 按 VIP 值从大到小排序
vip_df = vip_df.sort_values(by='VIP', ascending=False)

# 绘制 VIP 图
plt.figure(figsize=(8, 6))
sns.barplot(x='VIP', y='Feature', data=vip_df)
plt.title("Variable Importance in Projection (VIP) Plot (VIP > 1)")
plt.xlabel("VIP Value")
plt.ylabel("Features")
plt.show()