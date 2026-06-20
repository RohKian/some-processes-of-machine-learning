'''
        集成（esemble）是合并多个机器学习模型来构建更强大模型的方法。已证明有两种集成模型对大量
    分类和回归的数据集都是有效的，二者都以决策树为基础，分别是随机森林（random forest）和梯度提
    升决策树（gradient boosted decision tree）。
        决策树一个主要缺点在于经常对训练数据过拟合。随机森林是解决这个问题的一种方法。随机森林
    本质上是许多决策树的集合，其中每棵树都和其他树略有不同。
        随机森林背后的思想是：每棵树的预测可能都相对较好，但可能对部分数据过拟合。如果构造很多树，
    并且每棵树的预测都很好，但都以不同的方式过拟合，那么我们可以对这些树的结果。既能减少过拟合又能
    保持树的预测能力。
        为了实现这一策略，我们需要构造许多决策树。每棵树都应该对目标值做出可以接受的预测，还应该与
    其他树都各不相同。随机树中树的随机化方法有两种：
        1：通过选择构造树的数据点。
        2：通过选择每次划分测试的特征。

        构造随机森林
        想要构造一个随机森林模型，需要确定用于构造的树的个数（RandomForestRegressor 或 RandomForestClassifier
    的 n_estimators参数）。例如，我们想要构造 10 棵树。这些树在构造时彼此完全独立，算法对每棵树进行不同的随机
    选择，以确保树和树之间是有区别的。想要构造一棵树，
'''

# 一、随机森林分类器 —— RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import mglearn

# 解决 matplotlib 中文显示问题（Windows 上 DejaVu Sans 字体不含中文字符）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 加载乳腺癌数据集
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer['data'],
    cancer['target'],
    stratify=cancer['target'],
    random_state=42
)

# 1.1 单一决策树 vs 随机森林
print("=" * 60)
print("1.1 单一决策树 vs 随机森林（乳腺癌数据集）")
print("=" * 60)

# 单一决策树
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("单一决策树：")
print("  训练集精度：{:.3f}".format(tree.score(X_train, y_train)))
print("  测试集精度：{:.3f}".format(tree.score(X_test, y_test)))

# 随机森林（100棵树）
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)
print("\n随机森林（100棵树）：")
print("  训练集精度：{:.3f}".format(forest.score(X_train, y_train)))
print("  测试集精度：{:.3f}".format(forest.score(X_test, y_test)))

# 1.2 分析 n_estimators 对模型的影响
print("\n" + "=" * 60)
print("1.2 n_estimators（树的数量）对随机森林的影响")
print("=" * 60)

# 测试不同树数量对精度的影响
n_estimators_range = [1, 2, 3, 5, 10, 20, 50, 100, 200]
train_scores = []
test_scores = []

for n in n_estimators_range:
    forest = RandomForestClassifier(n_estimators=n, random_state=0, n_jobs=-1)
    forest.fit(X_train, y_train)
    train_scores.append(forest.score(X_train, y_train))
    test_scores.append(forest.score(X_test, y_test))

# 可视化
plt.figure(figsize=(10, 5))
plt.plot(n_estimators_range, train_scores, 'o-', label='训练集精度', color='blue')
plt.plot(n_estimators_range, test_scores, 's-', label='测试集精度', color='red')
plt.xlabel('n_estimators（树的数量）')
plt.ylabel('精度')
plt.title('随机森林：树的数量对精度的影响')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("关键发现：即使使用多棵树，随机森林也不会过拟合（训练集精度始终接近100%），")
print("          增加树的数量通常会使测试集精度趋于稳定。")

# 1.3 特征重要性分析
print("\n" + "=" * 60)
print("1.3 随机森林的特征重要性")
print("=" * 60)


def plot_feature_importance(model, feature_names, title="特征重要性"):
    """绘制特征重要性条形图"""
    n_features = len(feature_names)
    plt.figure(figsize=(10, 8))
    # 按重要性排序
    indices = np.argsort(model.feature_importances_)
    plt.barh(range(n_features),
             model.feature_importances_[indices],
             align='center')
    plt.yticks(np.arange(n_features),
               [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.tight_layout()
    plt.show()


plot_feature_importance(forest, cancer.feature_names, "随机森林特征重要性（乳腺癌数据集）")

# 对比：单一决策树的特征重要性
plot_feature_importance(tree, cancer.feature_names, "单一决策树特征重要性（乳腺癌数据集）")

print("随机森林的特征重要性更加稳定，不像单一决策树那样容易受个别划分的影响。")

# 二、随机森林回归器 —— RandomForestRegressor
print("\n" + "=" * 60)
print("二、随机森林回归（波士顿房价 / 加州房价数据集）")
print("=" * 60)

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score

# 加载加州房价数据集
housing = fetch_california_housing()
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    housing.data, housing.target, random_state=42
)

# 2.1 单一决策树回归 vs 随机森林回归
from sklearn.tree import DecisionTreeRegressor

# 单一决策树回归
tree_reg = DecisionTreeRegressor(random_state=0)
tree_reg.fit(X_train_h, y_train_h)
y_pred_tree = tree_reg.predict(X_test_h)

print("单一决策树回归：")
print("  R^2 得分（测试集）：{:.3f}".format(tree_reg.score(X_test_h, y_test_h)))
print("  MSE（测试集）：{:.3f}".format(mean_squared_error(y_test_h, y_pred_tree)))

# 随机森林回归
forest_reg = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
forest_reg.fit(X_train_h, y_train_h)
y_pred_forest = forest_reg.predict(X_test_h)

print("\n随机森林回归（100棵树）：")
print("  R^2 得分（测试集）：{:.3f}".format(forest_reg.score(X_test_h, y_test_h)))
print("  MSE（测试集）：{:.3f}".format(mean_squared_error(y_test_h, y_pred_forest)))

# 2.2 可视化预测结果
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test_h[:200], y_pred_tree[:200], alpha=0.5, s=10)
plt.plot([y_test_h.min(), y_test_h.max()],
         [y_test_h.min(), y_test_h.max()], 'r--', lw=2)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('决策树回归预测 vs 真实值（前200个样本）')

plt.subplot(1, 2, 2)
plt.scatter(y_test_h[:200], y_pred_forest[:200], alpha=0.5, s=10)
plt.plot([y_test_h.min(), y_test_h.max()],
         [y_test_h.min(), y_test_h.max()], 'r--', lw=2)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('随机森林回归预测 vs 真实值（前200个样本）')
plt.tight_layout()
plt.show()

# 三、梯度提升决策树（Gradient Boosted Decision Tree, GBDT）
print("\n" + "=" * 60)
print("三、梯度提升决策树 —— GradientBoostingClassifier")
print("=" * 60)

'''
    梯度提升（gradient boosting）是另一种集成方法，通过合并多个决策树来构建更强大的模型。

    与随机森林的不同之处：
    - 随机森林：所有树并行构造，相互独立，取平均值来降低过拟合
    - 梯度提升：树是顺序（串行）构造的，每棵树都尝试纠正前一棵树的错误

    默认情况下，梯度提升树没有随机化，而是用到了强预剪枝。梯度提升树使用
    深度很小（1到5之间）的树，这样模型占用的内存更少，预测速度也更快。

    梯度提升的主要思想是合并许多简单的模型（弱学习器），比如深度较小的树。
    每棵树只能对部分数据做出好的预测，因此，添加的树越来越多，可以不断迭代
    提高性能。

    梯度提升树经常是机器学习竞赛的优胜者，并且广泛应用于商业应用中。但它对
    参数设置更为敏感，如果参数设置不当，可能比随机森林更差。
'''

from sklearn.ensemble import GradientBoostingClassifier
# 3.1 基础梯度提升分类
print("3.1 梯度提升分类器（乳腺癌数据集）")

gbrt = GradientBoostingClassifier(
    max_depth=3,          # 树的深度限制（通常较小，1~5）
    n_estimators=100,     # 树的数量
    learning_rate=0.1,    # 学习率（控制每棵树对整体贡献的大小）
    random_state=0
)
gbrt.fit(X_train, y_train)

print("梯度提升分类器：")
print("  训练集精度：{:.3f}".format(gbrt.score(X_train, y_train)))
print("  测试集精度：{:.3f}".format(gbrt.score(X_test, y_test)))

# 3.2 学习率（learning_rate）的影响
print("\n3.2 学习率对梯度提升的影响")

# 不同学习率下，训练集和测试集的精度变化
learning_rates = [0.01, 0.05, 0.1, 0.5, 1.0]
for lr in learning_rates:
    gbrt_lr = GradientBoostingClassifier(
        max_depth=3,
        n_estimators=100,
        learning_rate=lr,
        random_state=0
    )
    gbrt_lr.fit(X_train, y_train)
    print("  learning_rate={:.2f}：训练集精度={:.3f}，测试集精度={:.3f}".format(
        lr, gbrt_lr.score(X_train, y_train), gbrt_lr.score(X_test, y_test)
    ))

print("\n说明：较小的学习率（如0.01）需要更多的树来构建更复杂的模型，")
print("      较大的学习率可能导致模型跳过最优解。")

# 3.3 n_estimators 对梯度提升的影响
print("\n3.3 n_estimators 对梯度提升的影响")

# 依次增加树的数量，观察精度变化
n_estimators_list = [1, 5, 10, 20, 50, 100, 200, 500]
train_scores_gb = []
test_scores_gb = []

for n in n_estimators_list:
    gbrt_n = GradientBoostingClassifier(
        max_depth=3,
        n_estimators=n,
        learning_rate=0.1,
        random_state=0
    )
    gbrt_n.fit(X_train, y_train)
    train_scores_gb.append(gbrt_n.score(X_train, y_train))
    test_scores_gb.append(gbrt_n.score(X_test, y_test))
    print("  n_estimators={:3d}：训练集精度={:.3f}，测试集精度={:.3f}".format(
        n, train_scores_gb[-1], test_scores_gb[-1]
    ))

plt.figure(figsize=(10, 5))
plt.plot(n_estimators_list, train_scores_gb, 'o-', label='训练集精度', color='blue')
plt.plot(n_estimators_list, test_scores_gb, 's-', label='测试集精度', color='red')
plt.xscale('log')
plt.xlabel('n_estimators（树的数量，对数坐标）')
plt.ylabel('精度')
plt.title('梯度提升：树的数量对精度的影响')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n注意：当n_estimators过大时，梯度提升可能出现过拟合（训练集精度持续上升，")
print("      但测试集精度开始下降），这与随机森林不同。")

# 3.4 按阶段（staged）查看预测精度
print("\n3.4 梯度提升的阶段性预测")

gbrt_staged = GradientBoostingClassifier(
    max_depth=3,
    n_estimators=100,
    learning_rate=0.1,
    random_state=0
)
gbrt_staged.fit(X_train, y_train)

# 使用 staged_predict 查看每添加一棵树后的预测结果
staged_scores_train = []
staged_scores_test = []
for y_pred_train_stage in gbrt_staged.staged_predict(X_train):
    staged_scores_train.append(np.mean(y_pred_train_stage == y_train))
for y_pred_test_stage in gbrt_staged.staged_predict(X_test):
    staged_scores_test.append(np.mean(y_pred_test_stage == y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(staged_scores_train) + 1), staged_scores_train,
         label='训练集精度', color='blue')
plt.plot(range(1, len(staged_scores_test) + 1), staged_scores_test,
         label='测试集精度', color='red')
plt.xlabel('n_estimators（累加的树的数量）')
plt.ylabel('精度')
plt.title('梯度提升：阶段性预测精度变化')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 四、梯度提升回归 —— GradientBoostingRegression
print("\n" + "=" * 60)
print("四、梯度提升回归 —— GradientBoostingRegressor")
print("=" * 60)

from sklearn.ensemble import GradientBoostingRegressor

gbrt_reg = GradientBoostingRegressor(
    max_depth=3,
    n_estimators=100,
    learning_rate=0.1,
    random_state=0
)
gbrt_reg.fit(X_train_h, y_train_h)
y_pred_gbrt = gbrt_reg.predict(X_test_h)

print("梯度提升回归：")
print("  R^2 得分（测试集）：{:.3f}".format(gbrt_reg.score(X_test_h, y_test_h)))
print("  MSE（测试集）：{:.3f}".format(mean_squared_error(y_test_h, y_pred_gbrt)))

# 对比：随机森林回归 vs 梯度提升回归
print("\n模型对比（回归任务）：")
print("  决策树        R^2 = {:.3f}".format(tree_reg.score(X_test_h, y_test_h)))
print("  随机森林      R^2 = {:.3f}".format(forest_reg.score(X_test_h, y_test_h)))
print("  梯度提升      R^2 = {:.3f}".format(gbrt_reg.score(X_test_h, y_test_h)))

# 五、网格搜索调参 —— GridSearchCV
print("\n" + "=" * 60)
print("五、网格搜索优化随机森林参数")
print("=" * 60)

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_split': [2, 5, 10]
}

# 注意：完整网格搜索比较耗时，这里只做演示
# 实际调参时可以根据需要缩小搜索范围
print("参数网格：")
print("  n_estimators:", param_grid['n_estimators'])
print("  max_depth:", param_grid['max_depth'])
print("  max_features:", param_grid['max_features'])
print("  min_samples_split:", param_grid['min_samples_split'])

# 使用较小的网格做快速演示
grid_small = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=0, n_jobs=-1),
    param_grid=grid_small,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("\n网格搜索结果：")
print("  最佳参数：", grid_search.best_params_)
print("  最佳交叉验证精度：{:.3f}".format(grid_search.best_score_))
print("  测试集精度：{:.3f}".format(grid_search.score(X_test, y_test)))
# 六、模型综合对比
print("\n" + "=" * 60)
print("六、模型综合对比")
print("=" * 60)

# 分类任务对比
models_clf = {
    '决策树': DecisionTreeClassifier(random_state=0),
    '随机森林（10棵树）': RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=-1),
    '随机森林（100棵树）': RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
    '梯度提升（0.1学习率）': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0),
}

print("分类任务对比（乳腺癌数据集）：")
print("-" * 60)
print("{:<30} {:>12} {:>12}".format("模型", "训练集精度", "测试集精度"))
print("-" * 60)
for name, model in models_clf.items():
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print("{:<30} {:>12.3f} {:>12.3f}".format(name, train_acc, test_acc))
print("-" * 60)

# 特征重要性对比
plt.figure(figsize=(12, 10))

# 随机森林特征重要性
plt.subplot(2, 2, 1)
indices_rf = np.argsort(forest.feature_importances_)
plt.barh(range(10), forest.feature_importances_[indices_rf[-10:]])
plt.yticks(range(10), [cancer.feature_names[i] for i in indices_rf[-10:]])
plt.title('随机森林 Top-10 特征重要性')

# 梯度提升特征重要性
plt.subplot(2, 2, 2)
indices_gb = np.argsort(gbrt.feature_importances_)
plt.barh(range(10), gbrt.feature_importances_[indices_gb[-10:]])
plt.yticks(range(10), [cancer.feature_names[i] for i in indices_gb[-10:]])
plt.title('梯度提升 Top-10 特征重要性')

# 随机森林 vs 梯度提升特征重要性散点图
plt.subplot(2, 2, (3, 4))
plt.scatter(forest.feature_importances_, gbrt.feature_importances_, alpha=0.6)
# 标注最重要的几个特征
top_indices = np.argsort(forest.feature_importances_ + gbrt.feature_importances_)[-5:]
for i in top_indices:
    plt.annotate(cancer.feature_names[i][:20],
                 (forest.feature_importances_[i], gbrt.feature_importances_[i]),
                 fontsize=8, alpha=0.8)
plt.plot([0, 0.5], [0, 0.5], 'r--', alpha=0.5)
plt.xlabel('随机森林特征重要性')
plt.ylabel('梯度提升特征重要性')
plt.title('随机森林 vs 梯度提升 特征重要性对比')

plt.tight_layout()
plt.show()

# 七、总结
print("\n" + "=" * 60)
print("七、总结")
print("=" * 60)

print("""
【随机森林（Random Forest）】
  优点：
    1. 几乎不需要调参就能获得不错的效果
    2. 不容易过拟合（增加树的数量不会导致过拟合）
    3. 可以给出特征重要性
    4. 对数据缩放不敏感（不需要归一化/标准化）
    5. 可以并行训练（各棵树独立构造）

  缺点：
    1. 模型较大，占用内存较多
    2. 预测速度相对较慢
    3. 对于高维稀疏数据（如文本数据）效果不好

【梯度提升决策树（Gradient Boosted Decision Tree）】
  优点：
    1. 通常比随机森林精度更高
    2. 可以处理混合类型特征
    3. 对数据缩放不敏感

  缺点：
    1. 需要仔细调参（学习率、树的数量、深度等）
    2. 训练时间长（顺序构造，无法并行）
    3. 对异常值敏感
    4. 如果参数设置不当，容易过拟合

【关键参数对比】
  随机森林的关键参数：
    - n_estimators：树的数量（越大越好，但收益递减）
    - max_features：每次划分考虑的特征数
    - max_depth：树的深度（通常不限制，靠集成防止过拟合）
    - bootstrap：是否使用自助采样（默认True）

  梯度提升的关键参数：
    - n_estimators：树的数量（需与学习率配合）
    - learning_rate：学习率（越小需要越多的树）
    - max_depth：树的深度（通常很小，1~5）
    - subsample：每次训练使用的样本比例（引入随机性）
""")
