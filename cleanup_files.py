#!/usr/bin/env python3
"""
생성된 파일들 정리 스크립트
"""

import os
import glob

def cleanup_files():
    """불필요한 파일들 정리"""
    print("=" * 50)
    print("파일 정리 시작")
    print("=" * 50)
    
    # 1. 제출용 파일들 확인
    submission_files = [
        "itransformer_submission.csv",
        "itransformer_cv_ensemble_submission.csv", 
        "itransformer_specialist_ensemble_submission.csv"
    ]
    
    print("📁 제출용 파일들:")
    for file in submission_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (없음)")
    
    # 2. 상세 결과 파일들 (분석용)
    detailed_files = [
        "itransformer_detailed.csv",
        "itransformer_cv_ensemble_detailed.csv",
        "itransformer_specialist_ensemble_detailed.csv"
    ]
    
    print("\n📊 상세 결과 파일들 (분석용):")
    for file in detailed_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (없음)")
    
    # 3. 모델 체크포인트들
    model_files = glob.glob("specialist_class_*.pth")
    cv_model_files = glob.glob("cv_models/*.pth")
    
    print(f"\n🤖 모델 체크포인트들:")
    print(f"  Specialist 모델: {len(model_files)}개")
    print(f"  CV 모델: {len(cv_model_files)}개")
    
    # 4. 정리 옵션
    print("\n" + "=" * 50)
    print("정리 옵션:")
    print("1. 제출용 파일만 남기고 나머지 삭제")
    print("2. 모든 파일 유지")
    print("3. 상세 결과 파일만 삭제")
    
    choice = input("\n선택하세요 (1-3): ").strip()
    
    if choice == "1":
        # 제출용 파일만 남기고 나머지 삭제
        files_to_delete = detailed_files + model_files + cv_model_files
        for file in files_to_delete:
            if os.path.exists(file):
                os.remove(file)
                print(f"삭제됨: {file}")
        print("✅ 제출용 파일만 남겼습니다.")
        
    elif choice == "3":
        # 상세 결과 파일만 삭제
        for file in detailed_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"삭제됨: {file}")
        print("✅ 상세 결과 파일을 삭제했습니다.")
    
    else:
        print("✅ 모든 파일을 유지합니다.")
    
    print("\n🎯 제출할 파일:")
    for file in submission_files:
        if os.path.exists(file):
            print(f"  📤 {file}")

if __name__ == "__main__":
    cleanup_files()
