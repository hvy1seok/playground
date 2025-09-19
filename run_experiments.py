#!/usr/bin/env python3
"""
TimesNet 실험 실행 스크립트
다양한 설정으로 실험을 쉽게 실행할 수 있습니다.
"""

import subprocess
import sys
import os

def run_experiment(name, **kwargs):
    """실험 실행"""
    cmd = [sys.executable, "timesnet_classification.py"]
    
    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))
    
    print(f"\n{'='*60}")
    print(f"실험: {name}")
    print(f"명령어: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ 실험 완료!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 실험 실패: {e}")
        print(f"오류 출력: {e.stderr}")
        return False

def main():
    """메인 함수"""
    print("TimesNet 실험 실행 스크립트")
    print("=" * 60)
    
    # 실험 결과 디렉토리 생성
    os.makedirs("experiments", exist_ok=True)
    
    # 기본 실험들
    experiments = [
        {
            "name": "기본 설정 (StandardScaler + Cosine)",
            "scaling": "standard",
            "scheduler": "cosine",
            "e_layers": 2,
            "d_model": 64,
            "d_ff": 128,
            "top_k": 5,
            "num_kernels": 6,
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 30,
            "patience": 10
        },
        {
            "name": "시계열 정규화 + Cosine",
            "scaling": "sequence",
            "scheduler": "cosine",
            "e_layers": 2,
            "d_model": 64,
            "d_ff": 128,
            "top_k": 5,
            "num_kernels": 6,
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 30,
            "patience": 10
        },
        {
            "name": "큰 모델 + Plateau 스케줄러",
            "scaling": "standard",
            "scheduler": "plateau",
            "e_layers": 3,
            "d_model": 96,
            "d_ff": 192,
            "top_k": 7,
            "num_kernels": 8,
            "learning_rate": 0.0005,
            "batch_size": 32,
            "epochs": 40,
            "patience": 15
        },
        {
            "name": "빠른 실험 (작은 모델)",
            "scaling": "standard",
            "scheduler": "cosine",
            "e_layers": 1,
            "d_model": 32,
            "d_ff": 64,
            "top_k": 3,
            "num_kernels": 4,
            "learning_rate": 0.002,
            "batch_size": 128,
            "epochs": 20,
            "patience": 5
        },
        {
            "name": "워밍업 코사인 스케줄러",
            "scaling": "standard",
            "scheduler": "warmup_cosine",
            "e_layers": 2,
            "d_model": 64,
            "d_ff": 128,
            "top_k": 5,
            "num_kernels": 6,
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 30,
            "patience": 10
        }
    ]
    
    # 사용자에게 실험 선택
    print("실행할 실험을 선택하세요:")
    print("0. 모든 실험 실행")
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp['name']}")
    
    choice = input("\n선택 (0-5): ").strip()
    
    if choice == "0":
        # 모든 실험 실행
        print("\n모든 실험을 순차적으로 실행합니다...")
        success_count = 0
        for exp in experiments:
            if run_experiment(exp["name"], **{k: v for k, v in exp.items() if k != "name"}):
                success_count += 1
        print(f"\n완료: {success_count}/{len(experiments)} 실험 성공")
    
    elif choice.isdigit() and 1 <= int(choice) <= len(experiments):
        # 선택된 실험 실행
        exp = experiments[int(choice) - 1]
        run_experiment(exp["name"], **{k: v for k, v in exp.items() if k != "name"})
    
    else:
        print("잘못된 선택입니다.")

if __name__ == "__main__":
    main()
