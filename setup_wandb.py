#!/usr/bin/env python3
"""
Wandb 설정 스크립트
Wandb 로그인 및 프로젝트 설정을 도와주는 스크립트입니다.
"""

import os
import wandb

def setup_wandb():
    """Wandb 설정"""
    print("Wandb 설정을 시작합니다...")
    
    # Wandb 로그인 확인
    try:
        if wandb.api.api_key is None:
            print("Wandb API 키가 설정되지 않았습니다.")
            print("다음 중 하나를 선택하세요:")
            print("1. wandb login 명령어 실행")
            print("2. 환경변수 WANDB_API_KEY 설정")
            print("3. 이 스크립트에서 직접 입력")
            
            choice = input("선택 (1/2/3): ").strip()
            
            if choice == "3":
                api_key = input("Wandb API 키를 입력하세요: ").strip()
                os.environ["WANDB_API_KEY"] = api_key
                print("API 키가 설정되었습니다.")
            else:
                print("선택한 방법으로 Wandb를 설정한 후 다시 실행하세요.")
                return False
        else:
            print("✓ Wandb API 키가 이미 설정되어 있습니다.")
    except Exception as e:
        print(f"Wandb 설정 중 오류 발생: {e}")
        return False
    
    # 프로젝트 설정
    project_name = input("프로젝트 이름을 입력하세요 (기본값: timesnet-classification): ").strip()
    if not project_name:
        project_name = "timesnet-classification"
    
    entity_name = input("Entity 이름을 입력하세요 (선택사항, 엔터로 건너뛰기): ").strip()
    if not entity_name:
        entity_name = None
    
    print(f"\n설정된 정보:")
    print(f"  프로젝트: {project_name}")
    print(f"  Entity: {entity_name if entity_name else '기본값 사용'}")
    
    # 테스트 실행
    try:
        run = wandb.init(
            project=project_name,
            entity=entity_name,
            config={"test": True},
            name="wandb_setup_test"
        )
        print("✓ Wandb 연결 테스트 성공!")
        wandb.finish()
        return True
    except Exception as e:
        print(f"✗ Wandb 연결 테스트 실패: {e}")
        return False

def main():
    """메인 함수"""
    print("=" * 50)
    print("Wandb 설정 도우미")
    print("=" * 50)
    
    if setup_wandb():
        print("\n✓ Wandb 설정이 완료되었습니다!")
        print("이제 timesnet_classification.py를 실행할 수 있습니다.")
    else:
        print("\n✗ Wandb 설정에 실패했습니다.")
        print("수동으로 Wandb를 설정한 후 다시 시도하세요.")

if __name__ == "__main__":
    main()
