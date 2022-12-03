# Face_Converter
졸업작품 - 자유주제

# <소개>
  - 2차원 인물 이미지를 3차원 이미지로 변환해주는 프로젝트
  - Django를 활용한 웹-클라이언트 시스템
  - 얼굴에 대한 3D 모델은 직접생성, 헤어에 대한 3D 모델은 라이브러리 매칭

 ## 시스템 흐름
  <img src="./readme_images/flow_diagram.png">
  
  **1) 이미지를 필터링**
  **2) 3D 얼굴 이미지 생성 : create_face**
  **3) 헤어 영역 추출 : hair_detect**
  **4) 유사한 헤어 매칭 : hair_similarity**
  **5) 합치기 및 출력**
  
  
 
## <개발환경>
- IDE : PyCharm
- Hardware : AMD Ryzen 7 5700x (3.4GHz, 8 cores), Nvidia GeForce RTX 3060ti, RAM 32GB

## <개발기간>
- 2022년 학기중 (6개월)
