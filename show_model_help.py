#!/usr/bin/env python3
"""
모델별 사용법 도움말 스크립트
"""

import subprocess
import sys

def show_help():
    """도움말 표시"""
    print("Time-Series-Library 모델 사용법")
    print("=" * 60)
    
    print("\n1. 사용 가능한 모델들:")
    print("   - TimesNet: FFT 기반 주기성 탐지 + 2D CNN")
    print("   - iTransformer: Inverted Transformer (변수별 어텐션)")
    print("   - PatchTST: Patch-based Time Series Transformer")
    
    print("\n2. 기본 사용법:")
    print("   # TimesNet")
    print("   python timesnet_classification.py --help")
    print("   python timesnet_classification.py --e_layers 4")
    
    print("\n   # iTransformer")
    print("   python itransformer_classification.py --help")
    print("   python itransformer_classification.py --e_layers 2")
    
    print("\n   # PatchTST")
    print("   python patchtst_classification.py --help")
    print("   python patchtst_classification.py --e_layers 3")
    
    print("\n3. 모델별 특징:")
    print("   TimesNet:")
    print("     - FFT로 주기성 자동 탐지")
    print("     - 2D CNN으로 패턴 학습")
    print("     - 시계열 데이터에 최적화")
    print("     - 권장 설정: e_layers=4, d_model=64")
    
    print("\n   iTransformer:")
    print("     - 변수별 어텐션 메커니즘")
    print("     - Inverted 구조로 효율성 향상")
    print("     - 다변량 시계열에 적합")
    print("     - 권장 설정: e_layers=2, d_model=64")
    
    print("\n   PatchTST:")
    print("     - 패치 기반 시계열 처리")
    print("     - Vision Transformer 스타일")
    print("     - 긴 시계열에 효과적")
    print("     - 권장 설정: e_layers=3, d_model=128, patch_len=16")
    
    print("\n4. 공통 옵션들:")
    print("   --scaling: 정규화 방법 (standard, minmax, robust, sequence, none)")
    print("   --scheduler: 스케줄러 (cosine, step, plateau, exponential, warmup_cosine)")
    print("   --e_layers: 레이어 수")
    print("   --d_model: 모델 차원")
    print("   --d_ff: Feed-forward 차원")
    print("   --learning_rate: 학습률")
    print("   --batch_size: 배치 크기")
    print("   --epochs: 학습 에포크 수")
    print("   --use_wandb: Wandb 로깅 사용")
    print("   --no_wandb: Wandb 로깅 비활성화")
    
    print("\n5. 모델별 특화 옵션:")
    print("   TimesNet:")
    print("     --top_k: FFT에서 선택할 상위 k개 주기 (default: 5)")
    print("     --num_kernels: Inception block의 커널 수 (default: 6)")
    
    print("\n   PatchTST:")
    print("     --patch_len: 패치 길이 (default: 16)")
    print("     --stride: 패치 스트라이드 (default: 8)")
    
    print("\n6. 사용 예시:")
    print("   # TimesNet 최적 설정")
    print("   python timesnet_classification.py --e_layers 4 --scaling standard --scheduler cosine")
    
    print("\n   # iTransformer 기본 설정")
    print("   python itransformer_classification.py --e_layers 2 --d_model 64")
    
    print("\n   # PatchTST 큰 모델")
    print("   python patchtst_classification.py --e_layers 4 --d_model 128 --patch_len 16")
    
    print("\n   # 모든 모델 비교 실험")
    print("   python compare_models.py")
    
    print("\n7. 정규화 방법 설명:")
    print("   standard: StandardScaler (FFT에 가장 적합, 권장)")
    print("   sequence: 시계열별 정규화 (각 샘플 독립 정규화)")
    print("   minmax: MinMaxScaler (0-1 정규화)")
    print("   robust: RobustScaler (중앙값 기반)")
    print("   none: 스케일링 없음 (원본 데이터)")
    
    print("\n8. 스케줄러 설명:")
    print("   cosine: 코사인 어닐링 (부드러운 감소, 권장)")
    print("   plateau: 플래토 스케줄러 (성능 개선 없으면 감소)")
    print("   warmup_cosine: 워밍업 + 코사인 (초기 워밍업 후 코사인)")
    print("   step: 스텝 스케줄러 (고정 간격으로 감소)")
    print("   exponential: 지수 감소 스케줄러")

if __name__ == "__main__":
    show_help()
