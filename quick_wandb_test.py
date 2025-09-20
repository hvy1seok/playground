#!/usr/bin/env python3
"""
빠른 Wandb 테스트
"""

import wandb
import time

def quick_test():
    print("Wandb 빠른 테스트 시작...")
    
    try:
        # Wandb run 시작
        run = wandb.init(
            project="quick-test",
            name=f"test-{int(time.time())}",
            config={"test": True}
        )
        
        # 메트릭 로깅
        for i in range(3):
            wandb.log({"test_metric": i * 0.5})
            print(f"메트릭 로깅: {i * 0.5}")
        
        print(f"✅ 성공! Wandb URL: {run.url}")
        wandb.finish()
        
    except Exception as e:
        print(f"❌ 실패: {e}")

if __name__ == "__main__":
    quick_test()
