# 🎯 유튜브 필터 버블 분석기

내가 시청한 유튜브 영상 링크를 입력하면, 카테고리 **다양성 지수**와 **샤논 엔트로피 기반 편향도(%)**를 계산해서 필터 버블 상태를 진단해주는 웹앱입니다.

## 사용한 지표

- **다양성 지수** = (등장한 고유 카테고리 수) / (전체 카테고리 수)
- **편향도(%)** = (1 - 정규화된 샤논 엔트로피) × 100
  - 샤논 엔트로피 공식: H(X) = -Σ P(xᵢ) log₂ P(xᵢ)
  - 편향도가 높을수록 특정 카테고리에 시청 콘텐츠가 쏠려있다는 의미

## 실행 방법 (로컬)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 필요한 것

- Gemini API 키 ([Google AI Studio](https://aistudio.google.com/apikey)에서 무료 발급)
- (선택) 유튜브 로그인 쿠키 파일 — 봇 차단 에러가 뜰 경우 필요

## 기술 스택

- Streamlit (웹 UI)
- yt-dlp (영상 메타데이터 수집)
- Gemini API (카테고리 분류)
- Matplotlib (시각화)
