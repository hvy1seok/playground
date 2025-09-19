#!/bin/bash

# TimesNet 정규화 방법 및 하이퍼파라미터 실험 스크립트
# Val Macro F1 기준으로 최적 파라미터 찾기

echo "=========================================="
echo "TimesNet 정규화 방법 및 하이퍼파라미터 실험"
echo "=========================================="

# 실험 결과 저장 디렉토리 생성
mkdir -p experiments
mkdir -p results

# 실험 시작 시간 기록
start_time=$(date +%s)
echo "실험 시작 시간: $(date)"

# 정규화 방법들
SCALING_METHODS=("standard" "minmax" "robust" "sequence")

# 하이퍼파라미터 조합들
declare -a PARAMS=(
    # 기본 설정
    "2 32 64 3 4 0.001 64"
    "2 64 128 5 6 0.001 64"
    "3 64 128 5 6 0.001 64"
    "2 128 256 5 6 0.001 64"
    
    # 학습률 변화
    "2 64 128 5 6 0.0005 64"
    "2 64 128 5 6 0.002 64"
    "2 64 128 5 6 0.005 64"
    
    # 배치 크기 변화
    "2 64 128 5 6 0.001 32"
    "2 64 128 5 6 0.001 128"
    "2 64 128 5 6 0.001 256"
    
    # 모델 크기 변화
    "1 32 64 3 4 0.001 64"
    "4 64 128 5 6 0.001 64"
    "2 32 64 3 4 0.001 32"
    "2 96 192 7 8 0.001 64"
)

# 결과 저장 파일
RESULTS_FILE="results/experiment_results.csv"
echo "scaling_method,e_layers,d_model,d_ff,top_k,num_kernels,learning_rate,batch_size,val_macro_f1,training_time,status" > $RESULTS_FILE

# 실험 카운터
experiment_count=0
total_experiments=$((${#SCALING_METHODS[@]} * ${#PARAMS[@]}))
echo "총 실험 수: $total_experiments"

# 각 정규화 방법에 대해 실험
for scaling_method in "${SCALING_METHODS[@]}"; do
    echo ""
    echo "=========================================="
    echo "정규화 방법: $scaling_method"
    echo "=========================================="
    
    # 각 하이퍼파라미터 조합에 대해 실험
    for param_string in "${PARAMS[@]}"; do
        experiment_count=$((experiment_count + 1))
        
        # 파라미터 파싱
        IFS=' ' read -r e_layers d_model d_ff top_k num_kernels learning_rate batch_size <<< "$param_string"
        
        echo ""
        echo "[$experiment_count/$total_experiments] 실험 중..."
        echo "정규화: $scaling_method"
        echo "파라미터: e_layers=$e_layers, d_model=$d_model, d_ff=$d_ff, top_k=$top_k, num_kernels=$num_kernels, lr=$learning_rate, batch_size=$batch_size"
        
        # 실험 시작 시간
        exp_start_time=$(date +%s)
        
        # Python 스크립트 실행
        python3 -c "
import sys
sys.path.append('./Time-Series-Library')
from timesnet_classification import main, TimesNetConfig, TimesNetTrainer
import pandas as pd
import numpy as np
import torch
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

def run_experiment(scaling_method, e_layers, d_model, d_ff, top_k, num_kernels, learning_rate, batch_size):
    try:
        # 데이터 로드
        train_df = pd.read_csv('datasests/train.csv')
        X = train_df.drop(columns=['ID', 'target']).values
        y = train_df['target'].values
        
        # 작은 샘플로 빠른 실험 (전체 데이터의 20%)
        sample_size = min(4000, len(X))
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
        
        # 학습/검증 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X_sample, y_sample, test_size=0.2, stratify=y_sample, random_state=123
        )
        
        # 정규화 적용
        if scaling_method == 'standard':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
        elif scaling_method == 'robust':
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
        elif scaling_method == 'sequence':
            def normalize_sequence(seq):
                seq_mean = np.mean(seq, axis=1, keepdims=True)
                seq_std = np.std(seq, axis=1, keepdims=True)
                return (seq - seq_mean) / (seq_std + 1e-8)
            X_train_scaled = normalize_sequence(X_train)
            X_val_scaled = normalize_sequence(X_val)
        
        # 시계열 변환
        X_train_ts = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
        X_val_ts = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1], 1)
        
        # 설정 생성
        config = TimesNetConfig()
        config.e_layers = e_layers
        config.d_model = d_model
        config.d_ff = d_ff
        config.top_k = top_k
        config.num_kernels = num_kernels
        config.learning_rate = learning_rate
        config.batch_size = batch_size
        config.train_epochs = 10  # 빠른 실험을 위해 10 에포크만
        config.patience = 5
        config.use_wandb = False  # 실험 중에는 Wandb 비활성화
        
        # 모델 생성
        model = TimesNet(config).to(config.device)
        optimizer = torch.optim.RAdam(model.parameters(), lr=config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        
        # DataLoader 생성
        train_mask = torch.ones(X_train_ts.shape[0], X_train_ts.shape[1])
        val_mask = torch.ones(X_val_ts.shape[0], X_val_ts.shape[1])
        
        train_dataset = TensorDataset(
            torch.tensor(X_train_ts, dtype=torch.float32),
            train_mask,
            torch.tensor(y_train, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_ts, dtype=torch.float32),
            val_mask,
            torch.tensor(y_val, dtype=torch.long)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        # 학습
        best_f1 = 0
        for epoch in range(config.train_epochs):
            # 학습
            model.train()
            for batch_x, batch_mask, batch_y in train_loader:
                batch_x = batch_x.to(config.device)
                batch_mask = batch_mask.to(config.device)
                batch_y = batch_y.to(config.device)
                
                optimizer.zero_grad()
                outputs = model.classification(batch_x, batch_mask)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # 검증
            model.eval()
            all_preds = []
            all_trues = []
            
            with torch.no_grad():
                for batch_x, batch_mask, batch_y in val_loader:
                    batch_x = batch_x.to(config.device)
                    batch_mask = batch_mask.to(config.device)
                    batch_y = batch_y.to(config.device)
                    
                    outputs = model.classification(batch_x, batch_mask)
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_trues.extend(batch_y.cpu().numpy())
            
            f1 = f1_score(all_trues, all_preds, average='macro')
            if f1 > best_f1:
                best_f1 = f1
        
        return best_f1, 'success'
        
    except Exception as e:
        print(f'오류 발생: {e}')
        return 0.0, 'error'

# 실험 실행
val_macro_f1, status = run_experiment('$scaling_method', $e_layers, $d_model, $d_ff, $top_k, $num_kernels, $learning_rate, $batch_size)

# 결과 출력
print(f'Val Macro F1: {val_macro_f1:.4f}')
print(f'Status: {status}')
" > temp_experiment.py
        
        # 실험 실행 및 결과 저장
        result=$(python3 temp_experiment.py 2>&1)
        val_macro_f1=$(echo "$result" | grep "Val Macro F1:" | awk '{print $4}')
        status=$(echo "$result" | grep "Status:" | awk '{print $2}')
        
        # 실험 종료 시간
        exp_end_time=$(date +%s)
        training_time=$((exp_end_time - exp_start_time))
        
        # 결과를 CSV에 저장
        echo "$scaling_method,$e_layers,$d_model,$d_ff,$top_k,$num_kernels,$learning_rate,$batch_size,$val_macro_f1,$training_time,$status" >> $RESULTS_FILE
        
        echo "결과: Val Macro F1 = $val_macro_f1, 시간 = ${training_time}초, 상태 = $status"
        
        # 임시 파일 삭제
        rm -f temp_experiment.py
    done
done

# 실험 종료 시간
end_time=$(date +%s)
total_time=$((end_time - start_time))

echo ""
echo "=========================================="
echo "실험 완료!"
echo "총 소요 시간: ${total_time}초 ($(($total_time / 60))분)"
echo "=========================================="

# 최고 성능 결과 분석
echo ""
echo "최고 성능 결과 분석:"
echo "===================="

# CSV 파일에서 최고 성능 찾기
best_result=$(tail -n +2 $RESULTS_FILE | sort -t',' -k9 -nr | head -n 1)
echo "최고 성능: $best_result"

# 상위 5개 결과 출력
echo ""
echo "상위 5개 결과:"
echo "순위,정규화방법,e_layers,d_model,d_ff,top_k,num_kernels,learning_rate,batch_size,val_macro_f1,training_time,status"
echo "1,$(tail -n +2 $RESULTS_FILE | sort -t',' -k9 -nr | head -n 1)"
echo "2,$(tail -n +2 $RESULTS_FILE | sort -t',' -k9 -nr | head -n 2 | tail -n 1)"
echo "3,$(tail -n +2 $RESULTS_FILE | sort -t',' -k9 -nr | head -n 3 | tail -n 1)"
echo "4,$(tail -n +2 $RESULTS_FILE | sort -t',' -k9 -nr | head -n 4 | tail -n 1)"
echo "5,$(tail -n +2 $RESULTS_FILE | sort -t',' -k9 -nr | head -n 5 | tail -n 1)"

# 정규화 방법별 평균 성능
echo ""
echo "정규화 방법별 평균 성능:"
echo "========================"
for scaling_method in "${SCALING_METHODS[@]}"; do
    avg_f1=$(tail -n +2 $RESULTS_FILE | grep "^$scaling_method," | awk -F',' '{sum+=$9; count++} END {if(count>0) print sum/count; else print 0}')
    count=$(tail -n +2 $RESULTS_FILE | grep "^$scaling_method," | wc -l)
    echo "$scaling_method: 평균 F1 = $avg_f1 (실험 수: $count)"
done

echo ""
echo "실험 결과가 $RESULTS_FILE에 저장되었습니다."
echo "최적 파라미터로 최종 모델을 학습하려면:"
echo "python3 timesnet_classification.py"
