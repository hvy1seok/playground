#!/usr/bin/env python3
"""
Wandb 연결 테스트 스크립트
"""

import wandb
import torch
import numpy as np

def test_wandb_connection():
    """Wandb 연결 테스트"""
    print("=" * 50)
    print("Wandb 연결 테스트")
    print("=" * 50)
    
    # 1. Wandb 버전 확인
    print(f"Wandb 버전: {wandb.__version__}")
    
    # 2. API 키 확인
    try:
        api_key = wandb.api.api_key
        if api_key:
            print(f"API 키 상태: 연결됨 (길이: {len(api_key)})")
        else:
            print("API 키 상태: 없음")
    except Exception as e:
        print(f"API 키 확인 실패: {e}")
    
    # 3. 간단한 Wandb run 테스트
    try:
        print("\nWandb run 테스트 중...")
        run = wandb.init(
            project="wandb-test",
            name="connection-test",
            config={"test": True},
            reinit=True
        )
        
        # 간단한 메트릭 로깅
        wandb.log({"test_metric": 1.0})
        print("✅ Wandb 로깅 성공!")
        
        # Run 종료
        wandb.finish()
        print("✅ Wandb run 종료 성공!")
        
    except Exception as e:
        print(f"❌ Wandb 테스트 실패: {e}")
        print("\n해결 방법:")
        print("1. wandb login 명령어 실행")
        print("2. 인터넷 연결 확인")
        print("3. Wandb 계정 확인")

if __name__ == "__main__":
    test_wandb_connection()
