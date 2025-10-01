#!/usr/bin/env python3
"""
개선된 최종 앙상블 스크립트
실제 학습된 TabTransformer + iTransformer 결과 결합
"""

import pandas as pd
import numpy as np
import os

def main():
    print("개선된 TabTransformer + iTransformer 최종 앙상블")
    print("=" * 60)
    
    # 1. iTransformer Specialist 앙상블 결과 로드
    if os.path.exists("itransformer_specialist_ensemble_detailed.csv"):
        print("✅ iTransformer Specialist 앙상블 결과 로드")
        itransformer_df = pd.read_csv("itransformer_specialist_ensemble_detailed.csv")
        itransformer_probs = itransformer_df[[f'prob_{i}' for i in range(21)]].values
        itransformer_preds = itransformer_df['target'].values
        test_ids = itransformer_df['ID'].values
        print(f"iTransformer 데이터 형태: {itransformer_probs.shape}")
    else:
        print("❌ iTransformer Specialist 앙상블 결과를 찾을 수 없습니다.")
        return
    
    # 2. TabTransformer 5폴드 결과 로드
    if os.path.exists("tabtransformer_5fold_detailed.csv"):
        print("✅ TabTransformer 5폴드 결과 로드")
        tabtransformer_df = pd.read_csv("tabtransformer_5fold_detailed.csv")
        tabtransformer_probs = tabtransformer_df[[f'prob_{i}' for i in range(21)]].values
        tabtransformer_preds = tabtransformer_df['target'].values
        print(f"TabTransformer 데이터 형태: {tabtransformer_probs.shape}")
    else:
        print("❌ TabTransformer 5폴드 결과를 찾을 수 없습니다.")
        print("먼저 run_tabtransformer.py를 실행하세요.")
        return
    
    # 3. 모델별 예측 분포 분석
    print(f"\n모델별 예측 분포:")
    print(f"TabTransformer: {np.bincount(tabtransformer_preds, minlength=21)}")
    print(f"iTransformer: {np.bincount(itransformer_preds, minlength=21)}")
    
    # 4. 모델 간 일치도 분석
    agreement = np.mean(tabtransformer_preds == itransformer_preds)
    print(f"\n모델 간 예측 일치도: {agreement:.4f}")
    
    # 5. 확률 분포 분석 (엔트로피)
    tab_entropy = -np.sum(tabtransformer_probs * np.log(tabtransformer_probs + 1e-8), axis=1).mean()
    itransformer_entropy = -np.sum(itransformer_probs * np.log(itransformer_probs + 1e-8), axis=1).mean()
    
    print(f"\n평균 엔트로피 (불확실성):")
    print(f"TabTransformer: {tab_entropy:.4f}")
    print(f"iTransformer: {itransformer_entropy:.4f}")
    
    # 6. 다양한 앙상블 방법 실험
    print(f"\n앙상블 방법 실험:")
    
    ensemble_methods = [
        ("동일 가중치", 0.5, 0.5),
        ("iTransformer 우세", 0.3, 0.7),
        ("TabTransformer 우세", 0.7, 0.3),
        ("엔트로피 기반", None, None),  # 엔트로피 역비례 가중치
        ("일치도 기반", None, None),   # 일치도 기반 가중치
    ]
    
    best_method = None
    best_entropy = float('inf')
    best_agreement = 0
    
    for method_name, tab_weight, itransformer_weight in ensemble_methods:
        if method_name == "엔트로피 기반":
            # 엔트로피가 낮을수록 높은 가중치
            total_entropy = tab_entropy + itransformer_entropy
            tab_weight = (total_entropy - tab_entropy) / total_entropy
            itransformer_weight = (total_entropy - itransformer_entropy) / total_entropy
        elif method_name == "일치도 기반":
            # 일치도가 높을수록 높은 가중치 (단순화)
            tab_weight = 0.4
            itransformer_weight = 0.6
        else:
            # 고정 가중치 사용
            pass
        
        # 앙상블 예측
        final_probs = (tab_weight * tabtransformer_probs + 
                       itransformer_weight * itransformer_probs)
        final_preds = np.argmax(final_probs, axis=1)
        
        # 평가 지표
        final_entropy = -np.sum(final_probs * np.log(final_probs + 1e-8), axis=1).mean()
        tab_final_agreement = np.mean(tabtransformer_preds == final_preds)
        itransformer_final_agreement = np.mean(itransformer_preds == final_preds)
        avg_agreement = (tab_final_agreement + itransformer_final_agreement) / 2
        
        print(f"{method_name:15s}: 가중치({tab_weight:.2f}, {itransformer_weight:.2f}) | "
              f"엔트로피={final_entropy:.4f} | 평균일치도={avg_agreement:.4f}")
        
        # 최적 방법 선택 (엔트로피가 낮고 일치도가 높은 것)
        if final_entropy < best_entropy and avg_agreement > best_agreement:
            best_entropy = final_entropy
            best_agreement = avg_agreement
            best_method = (method_name, tab_weight, itransformer_weight, final_probs, final_preds)
    
    # 7. 최적 앙상블 결과 생성
    if best_method:
        method_name, tab_weight, itransformer_weight, final_probs, final_preds = best_method
        print(f"\n최적 앙상블 방법: {method_name}")
        print(f"최적 가중치: TabTransformer={tab_weight:.3f}, iTransformer={itransformer_weight:.3f}")
    else:
        # 기본값 사용
        tab_weight, itransformer_weight = 0.5, 0.5
        final_probs = (tab_weight * tabtransformer_probs + 
                       itransformer_weight * itransformer_probs)
        final_preds = np.argmax(final_probs, axis=1)
        print(f"\n기본 앙상블 사용: 가중치({tab_weight}, {itransformer_weight})")
    
    # 8. 최종 결과 저장
    final_submission = pd.DataFrame({
        "ID": test_ids,
        "target": final_preds
    })
    final_submission.to_csv("improved_ensemble_submission.csv", index=False)
    
    final_detailed = pd.DataFrame({
        "ID": test_ids,
        "target": final_preds,
        **{f"prob_{i}": final_probs[:, i] for i in range(21)}
    })
    final_detailed.to_csv("improved_ensemble_detailed.csv", index=False)
    
    # 9. 최종 결과 분석
    print(f"\n최종 앙상블 결과:")
    print(f"예측 분포: {np.bincount(final_preds, minlength=21)}")
    print(f"TabTransformer vs Final 일치도: {np.mean(tabtransformer_preds == final_preds):.4f}")
    print(f"iTransformer vs Final 일치도: {np.mean(itransformer_preds == final_preds):.4f}")
    print(f"최종 엔트로피: {-np.sum(final_probs * np.log(final_probs + 1e-8), axis=1).mean():.4f}")
    print(f"\n제출 파일: improved_ensemble_submission.csv")
    print(f"상세 결과: improved_ensemble_detailed.csv")

if __name__ == "__main__":
    main()
