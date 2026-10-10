# 🦷 Dentphoto AI Document Translator
> **Google AI Studio(Gemini API) 무료 플랜을 활용한 치의학 및 학술 문서 AI 번역기**

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![Google Gemini API](https://img.shields.io/badge/AI-Google%20Gemini-green.svg)](https://aistudio.google.com/)

**Dentphoto AI Document Translator**는 치과의사 및 연구자분들이 원서, 학술 논문, 해외 보도자료, 전문 서적(PDF, EPUB, TXT)을 매끄럽고 정확하게 번역할 수 있도록 개발된 웹 기반 AI 문서 번역 프로그램입니다.

Google AI Studio에서 제공하는 **무료 API 키**를 활용하여 비용 부담 없이 강력한 Gemini AI 모델로 긴 문서를 안전하게 번역할 수 있습니다.

---

## ✨ 핵심 기능

* 💸 **100% 무료 API 활용**: Google AI Studio의 Gemini 무료 API 키를 등록하여 무료로 사용 가능
* 📚 **다양한 포맷 지원**: `PDF`, `EPUB`, `TXT` 등 대용량 문서 자동 텍스트 추출 및 번역
* 📖 **원문+번역문 대역(Bilingual) 출력**:
  * `원문(번역문)` 형태의 대역본 파일 자동 생성
  * 순수 번역본과 대역본을 동시에 제공하여 학술 검토 용이
* 💾 **체크포인트(중단 시 이어서 번역) 지원**:
  * 번역 도중 네트워크가 끊기거나 오류가 발생해도 SQLite 기반 체크포인터로 **마지막 중단 지점부터 이어서 번역**
* 🔒 **보안 및 프라이버시**:
  * API 키는 환경변수 또는 메모리 상에서만 처리하며 로컬 로그에 절대 저장되지 않음
  * 임시파일 및 번역 결과물 개별/전체 삭제 관리 기능 포함
* 다수 파일 번역시 압축 기능 포함

---

## 🚀 시작하기

### 1. Google AI Studio API 키 발급
1. [Google AI Studio](https://aistudio.google.com/)에 접속하여 구글 계정으로 로그인합니다.
2. **Get API key**를 클릭하여 무료 API 키를 생성하고 복사합니다.

### 2. 설치 및 실행 (Linux / Lubuntu)

터미널을 열고 시작 스크립트를 실행하면 필요한 파이썬 가상환경(venv) 및 라이브러리가 자동으로 설치되고 웹 UI가 열립니다.

```bash
chmod +x start_lubuntu.sh
./start_lubuntu.sh
```

## 스스로 가상환경을 구성하여 실행하고 싶다면:

# 필수 패키지 설치
```bash
pip install -r requirements.txt
```

# 프로그램 실행
```bash
python3 dentphoto.py
```

실행 후 웹 브라우저에서 http://127.0.0.1:7860 주소로 접속합니다.

## 💡 사용 방법
API 키 입력: 발급받은 Google AI Studio API 키를 입력합니다. (환경변수 GEMINI_API_KEY 설정 시 자동 로드)

모델 선택: 모델 목록 조회 버튼을 눌러 최신 Gemini 모델을 선택 후 연결 테스트를 진행합니다.

파일 첨부: 번역할 문서(PDF, EPUB, TXT)를 업로드합니다.

번역 설정: 목표 언어, 문서 장르, 문체, 용어집(선택)을 설정합니다.

번역 시작: 번역 시작 / 이어서 번역 버튼을 클릭합니다.

결과 다운로드: 번역이 완료되면 개별 결과 파일 또는 전체 결과 ZIP 파일로 다운로드합니다.

## 🛠️ 요구 사항 (Prerequisites)
  * Python 3.11 이상
  * PyMuPDF (fitz) — PDF extraction
  * Gradio 4.0+ — Web User Interface
  * BeautifulSoup4 — EPUB document processing
  * Requests — Google Gemini REST API Communication

구글의 자원을 이용하기 때문에 로컬PC의 성능은 별로 중요하지 않습니다.(GPU 없는 내장그래픽에서도 구동 가능)


## 📌 주의사항
스캔된 이미지 형태의 PDF는 OCR 처리가 되어있지 않을 경우 텍스트 추출이 불가능할 수 있습니다.

API 무료 플랜 제한(RPM/TPM)에 맞춰 고급 설정에서 청크 크기와 요청 간격을 조절할 수 있습니다.
