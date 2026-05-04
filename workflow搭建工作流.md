# Workflow 工作流搭建指南

## 目录
1. [什么是 Workflow](#什么是-workflow)
2. [GitHub Actions 工作流搭建](#github-actions-工作流搭建)
3. [Apache Airflow 工作流搭建](#apache-airflow-工作流搭建)
4. [机器学习项目中的工作流](#机器学习项目中的工作流)
5. [最佳实践](#最佳实践)

---

## 什么是 Workflow

工作流（Workflow）是指一系列相互关联的任务按照特定的顺序和规则自动执行的过程。在软件开发和机器学习领域，工作流主要用于：

- **自动化重复性任务**
- **确保流程标准化**
- **提高团队协作效率**
- **实现持续集成/持续部署（CI/CD）**

---

## GitHub Actions 工作流搭建

### 1. 基础结构

在项目根目录创建 `.github/workflows/` 文件夹，并添加 YAML 配置文件：

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    # 每天凌晨2点运行
    - cron: '0 2 * * *'

jobs:
  train-model:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run data preprocessing
      run: python src/data_preprocessing.py
    
    - name: Train model
      run: python src/train.py
    
    - name: Evaluate model
      run: python src/evaluate.py
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: model-artifacts
        path: |
          models/
          reports/
```

### 2. 矩阵构建策略

```yaml
strategy:
  matrix:
    python-version: ['3.8', '3.9', '3.10']
    os: [ubuntu-latest, windows-latest]

steps:
- name: Test on ${{ matrix.os }} with Python ${{ matrix.python-version }}
  run: python -m pytest tests/
```

### 3. 环境变量与密钥

```yaml
env:
  MODEL_NAME: "random_forest"
  DATA_PATH: "./data"

jobs:
  deploy:
    steps:
    - name: Deploy to cloud
      env:
        API_KEY: ${{ secrets.API_KEY }}
        AWS_ACCESS_KEY: ${{ secrets.AWS_ACCESS_KEY }}
      run: python src/deploy.py
```

---

## Apache Airflow 工作流搭建

### 1. 安装与配置

```bash
# 安装 Airflow
pip install apache-airflow

# 初始化数据库
airflow db init

# 创建用户
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# 启动服务
airflow webserver --port 8080
airflow scheduler
```

### 2. 定义 DAG（有向无环图）

```python
# dags/ml_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# 默认参数
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': ['ml-team@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# 创建 DAG
dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='机器学习模型训练工作流',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'training'],
)

# 任务函数
def extract_data():
    """数据提取"""
    print("从数据库提取数据...")
    # 数据提取逻辑

def preprocess_data():
    """数据预处理"""
    print("数据清洗和特征工程...")
    # 预处理逻辑

def train_model():
    """模型训练"""
    print("训练机器学习模型...")
    # 训练逻辑

def evaluate_model():
    """模型评估"""
    print("评估模型性能...")
    # 评估逻辑

def deploy_model():
    """模型部署"""
    print("部署模型到生产环境...")
    # 部署逻辑

# 定义任务
t1 = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

t2 = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

t3 = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

t4 = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

t5 = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)

# 设置依赖关系
t1 >> t2 >> t3 >> t4 >> t5
```

### 3. 传感器与高级特性

```python
from airflow.sensors.filesystem import FileSensor
from airflow.providers.http.sensors.http import HttpSensor

# 文件传感器 - 等待文件出现
wait_for_data = FileSensor(
    task_id='wait_for_data',
    filepath='/data/new_data.csv',
    poke_interval=300,  # 每5分钟检查一次
    timeout=3600,       # 1小时超时
    dag=dag,
)

# HTTP 传感器 - 等待 API 可用
wait_for_api = HttpSensor(
    task_id='wait_for_api',
    http_conn_id='ml_api',
    endpoint='/health',
    poke_interval=60,
    dag=dag,
)
```

---

## 机器学习项目中的工作流

### 1. MLOps 完整工作流

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  数据收集    │ -> │  数据验证    │ -> │  数据转换    │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
┌─────────────┐    ┌─────────────┐    ┌─────▼───────┐
│  模型监控    │ <- │  模型部署    │ <- │  模型训练    │
└─────────────┘    └─────────────┘    └─────────────┘
       ^                                       │
       └───────────────────────────────────────┘
                    持续反馈
```

### 2. MLflow 集成工作流

```yaml
name: MLflow Tracking

jobs:
  experiment:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup MLflow
      run: pip install mlflow
    
    - name: Run experiment
      env:
        MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_URI }}
      run: |
        python -c "
        import mlflow
        import mlflow.sklearn
        from sklearn.ensemble import RandomForestClassifier
        
        mlflow.set_experiment('iris-classification')
        
        with mlflow.start_run():
            # 记录参数
            mlflow.log_param('n_estimators', 100)
            mlflow.log_param('max_depth', 5)
            
            # 训练模型
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_train, y_train)
            
            # 记录指标
            accuracy = model.score(X_test, y_test)
            mlflow.log_metric('accuracy', accuracy)
            
            # 保存模型
            mlflow.sklearn.log_model(model, 'model')
        "
```

### 3. 自动化特征工程工作流

```python
# feature_pipeline.py
import pandas as pd
from feature_engine.imputation import MeanMedianImputer
from feature_engine.encoding import OneHotEncoder
from feature_engine.selection import SelectByShuffling

def feature_engineering_pipeline():
    """自动化特征工程流程"""
    
    # 1. 数据加载
    data = pd.read_csv('raw_data.csv')
    
    # 2. 缺失值处理
    imputer = MeanMedianImputer(variables=['age', 'income'])
    data = imputer.fit_transform(data)
    
    # 3. 类别编码
    encoder = OneHotEncoder(variables=['category', 'region'])
    data = encoder.fit_transform(data)
    
    # 4. 特征选择
    selector = SelectByShuffling(
        estimator=RandomForestClassifier(),
        scoring='roc_auc',
        cv=3
    )
    data = selector.fit_transform(data, target)
    
    # 5. 保存处理后的数据
    data.to_csv('processed_data.csv', index=False)
    
    return data
```

---

## 最佳实践

### 1. 版本控制

```yaml
# 确保可重复性
- name: Save environment
  run: |
    pip freeze > requirements.txt
    python --version > python_version.txt
    
- name: Cache dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
```

### 2. 并行执行

```yaml
jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        test-group: [unit, integration, e2e]
    steps:
    - name: Run ${{ matrix.test-group }} tests
      run: pytest tests/${{ matrix.test-group }}/
```

### 3. 条件执行

```yaml
- name: Deploy to production
  if: github.ref == 'refs/heads/main' && github.event_name == 'push'
  run: python deploy.py --env=production

- name: Deploy to staging
  if: github.ref == 'refs/heads/develop'
  run: python deploy.py --env=staging
```

### 4. 错误处理与通知

```python
from airflow.utils.email import send_email

def notify_failure(context):
    """任务失败通知"""
    task_instance = context['task_instance']
    send_email(
        to=['team@company.com'],
        subject=f"任务失败: {task_instance.task_id}",
        html_content=f"""
        <h3>任务执行失败</h3>
        <p>DAG: {task_instance.dag_id}</p>
        <p>任务: {task_instance.task_id}</p>
        <p>执行时间: {context['execution_date']}</p>
        <p>日志: {task_instance.log_url}</p>
        """
    )

# 在 DAG 中使用
task = PythonOperator(
    task_id='risky_task',
    python_callable=risky_function,
    on_failure_callback=notify_failure,
    dag=dag,
)
```

### 5. 资源管理

```yaml
jobs:
  gpu-training:
    runs-on: self-hosted
    container:
      image: pytorch/pytorch:latest
      options: --gpus all
    steps:
    - name: Train with GPU
      run: python train_gpu.py
    
    - name: Cleanup
      if: always()
      run: |
        rm -rf /tmp/training_cache
        docker system prune -f
```

---

## 总结

| 工具 | 适用场景 | 学习曲线 | 扩展性 |
|------|---------|---------|--------|
| GitHub Actions | CI/CD, 自动化测试 | 低 | 中 |
| Apache Airflow | 复杂数据处理流程 | 中 | 高 |
| Prefect | 现代数据流程 | 低 | 高 |
| Kubeflow | Kubernetes 上的 ML | 高 | 高 |
| MLflow | 实验跟踪, 模型管理 | 低 | 中 |

选择合适的工作流工具，可以显著提升机器学习项目的开发效率和可维护性。