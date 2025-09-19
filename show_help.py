#!/usr/bin/env python3
"""
TimesNet 사용법 도움말 스크립트
"""

import subprocess
import sys

def show_help():
    """도움말 표시"""
    print("TimesNet 분류 모델 사용법")
    print("=" * 60)
    
    # 기본 도움말
    print("\n1. 기본 사용법:")
    print("   python timesnet_classification.py --help")
    
    # 주요 옵션들
    print("\n2. 주요 옵션들:")
    print("   --scaling: 정규화 방법 (standard, minmax, robust, sequence)")
    print("   --scheduler: 스케줄러 (cosine, step, plateau, exponential, warmup_cosine)")
    print("   --e_layers: TimesBlock 레이어 수 (1-4)")
    print("   --d_model: 모델 차원 (32, 64, 96, 128)")
    print("   --d_ff: Feed-forward 차원 (64, 128, 192, 256)")
    print("   --learning_rate: 학습률 (0.0001-0.01)")
    print("   --batch_size: 배치 크기 (16, 32, 64, 128, 256)")
    print("   --epochs: 학습 에포크 수 (10-100)")
    print("   --use_wandb: Wandb 로깅 사용")
    print("   --no_wandb: Wandb 로깅 비활성화")
    
    # 예시들
    print("\n3. 사용 예시:")
    print("   # 기본 설정")
    print("   python timesnet_classification.py")
    
    print("\n   # FFT에 최적화된 설정")
    print("   python timesnet_classification.py --scaling standard --scheduler cosine")
    
    print("\n   # 시계열별 정규화 + 플래토 스케줄러")
    print("   python timesnet_classification.py --scaling sequence --scheduler plateau")
    
    print("\n   # 큰 모델로 학습")
    print("   python timesnet_classification.py --e_layers 3 --d_model 96 --d_ff 192 --top_k 7")
    
    print("\n   # 빠른 실험")
    print("   python timesnet_classification.py --e_layers 1 --d_model 32 --epochs 20")
    
    print("\n   # Wandb 없이 학습")
    print("   python timesnet_classification.py --no_wandb")
    
    # 정규화 방법 설명
    print("\n4. 정규화 방법 설명:")
    print("   standard: StandardScaler (FFT에 가장 적합, 권장)")
    print("   sequence: 시계열별 정규화 (각 샘플 독립 정규화)")
    print("   minmax: MinMaxScaler (0-1 정규화)")
    print("   robust: RobustScaler (중앙값 기반, 기존 방식)")
    
    # 스케줄러 설명
    print("\n5. 스케줄러 설명:")
    print("   cosine: 코사인 어닐링 (부드러운 감소, 권장)")
    print("   plateau: 플래토 스케줄러 (성능 개선 없으면 감소)")
    print("   warmup_cosine: 워밍업 + 코사인 (초기 워밍업 후 코사인)")
    print("   step: 스텝 스케줄러 (고정 간격으로 감소)")
    print("   exponential: 지수 감소 스케줄러")
    
    # 실험 스크립트
    print("\n6. 실험 실행:")
    print("   python run_experiments.py  # 미리 정의된 실험들")
    print("   python test_schedulers.py  # 스케줄러 비교")
    print("   python run_hyperparameter_experiment.py  # 하이퍼파라미터 실험")
    
    print("\n7. 도움말 보기:")
    print("   python timesnet_classification.py --help")

if __name__ == "__main__":
    show_help()
