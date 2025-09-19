#!/usr/bin/env python3
"""
TimesNet, iTransformer, PatchTST 모델 비교 실험 스크립트
"""

import subprocess
import sys
import os
import time
import pandas as pd
from datetime import datetime

def run_experiment(model_name, **kwargs):
    """실험 실행"""
    if model_name == 'timesnet':
        cmd = [sys.executable, "timesnet_classification.py"]
    elif model_name == 'itransformer':
        cmd = [sys.executable, "itransformer_classification.py"]
    elif model_name == 'patchtst':
        cmd = [sys.executable, "patchtst_classification.py"]
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))
    
    print(f"\n{'='*60}")
    print(f"실험: {model_name.upper()}")
    print(f"명령어: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        end_time = time.time()
        training_time = end_time - start_time
        
        print("✅ 실험 완료!")
        print(f"실행 시간: {training_time:.2f}초")
        return True, training_time
    except subprocess.CalledProcessError as e:
        print(f"❌ 실험 실패: {e}")
        print(f"오류 출력: {e.stderr}")
        return False, 0

def main():
    """메인 함수"""
    print("TimesNet vs iTransformer vs PatchTST 모델 비교 실험")
    print("=" * 60)
    
    # 실험 결과 저장
    results = []
    
    # 기본 설정들
    experiments = [
        {
            "name": "기본 설정 비교",
            "models": [
                ("timesnet", {"e_layers": 4, "scaling": "standard", "scheduler": "cosine"}),
                ("itransformer", {"e_layers": 2, "scaling": "standard", "scheduler": "cosine"}),
                ("patchtst", {"e_layers": 3, "scaling": "standard", "scheduler": "cosine"})
            ]
        },
        {
            "name": "큰 모델 비교",
            "models": [
                ("timesnet", {"e_layers": 4, "d_model": 96, "d_ff": 192, "scaling": "standard", "scheduler": "cosine"}),
                ("itransformer", {"e_layers": 3, "d_model": 96, "d_ff": 192, "scaling": "standard", "scheduler": "cosine"}),
                ("patchtst", {"e_layers": 4, "d_model": 128, "d_ff": 256, "scaling": "standard", "scheduler": "cosine"})
            ]
        },
        {
            "name": "빠른 실험 (작은 모델)",
            "models": [
                ("timesnet", {"e_layers": 2, "d_model": 32, "d_ff": 64, "epochs": 20, "scaling": "standard", "scheduler": "cosine"}),
                ("itransformer", {"e_layers": 1, "d_model": 32, "d_ff": 64, "epochs": 20, "scaling": "standard", "scheduler": "cosine"}),
                ("patchtst", {"e_layers": 2, "d_model": 64, "d_ff": 128, "epochs": 20, "scaling": "standard", "scheduler": "cosine"})
            ]
        }
    ]
    
    # 사용자에게 실험 선택
    print("실행할 실험을 선택하세요:")
    print("0. 모든 실험 실행")
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp['name']}")
    
    choice = input("\n선택 (0-3): ").strip()
    
    if choice == "0":
        # 모든 실험 실행
        print("\n모든 실험을 순차적으로 실행합니다...")
        for exp in experiments:
            print(f"\n{'='*80}")
            print(f"실험: {exp['name']}")
            print(f"{'='*80}")
            
            for model_name, config in exp['models']:
                success, training_time = run_experiment(model_name, **config)
                results.append({
                    'experiment': exp['name'],
                    'model': model_name,
                    'config': str(config),
                    'success': success,
                    'training_time': training_time,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
    
    elif choice.isdigit() and 1 <= int(choice) <= len(experiments):
        # 선택된 실험 실행
        exp = experiments[int(choice) - 1]
        print(f"\n실험: {exp['name']}")
        
        for model_name, config in exp['models']:
            success, training_time = run_experiment(model_name, **config)
            results.append({
                'experiment': exp['name'],
                'model': model_name,
                'config': str(config),
                'success': success,
                'training_time': training_time,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    
    else:
        print("잘못된 선택입니다.")
        return
    
    # 결과 저장
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('model_comparison_results.csv', index=False)
        print(f"\n실험 결과가 'model_comparison_results.csv'에 저장되었습니다.")
        
        # 결과 요약
        print(f"\n{'='*60}")
        print("실험 결과 요약")
        print(f"{'='*60}")
        
        for _, row in results_df.iterrows():
            status = "✅ 성공" if row['success'] else "❌ 실패"
            print(f"{row['model']:12} | {status} | {row['training_time']:8.2f}초 | {row['experiment']}")
    
    print(f"\n총 {len(results)}개 실험 완료")

if __name__ == "__main__":
    main()
