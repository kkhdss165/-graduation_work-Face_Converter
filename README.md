# Face_Converter
졸업작품 - 자유주제

# <소개>
  - 2차원 인물 이미지를 3차원 이미지로 변환해주는 프로젝트
  - Django를 활용한 웹-클라이언트 시스템
  - 얼굴에 대한 3D 모델은 직접생성, 헤어에 대한 3D 모델은 라이브러리 매칭

 ## 시스템 흐름
  <img src="./readme_images/flow_diagram.png">
  
  1) 이미지를 필터링
  2) 3D 얼굴 이미지 생성 : create_face
  3) 헤어 영역 추출 : hair_detect
  4) 유사한 헤어 매칭 : hair_similarity
  5) 합치기 및 출력
  
  ### 1. 이미지 필터링
  
  <img src="./readme_images/dlib.png">
  - 얼굴 인식 라이브러리 dlib 활용하여 이미지속 인물이 1명이 아닌경우 필터링

  ### 2. create_face : 3D 얼굴 이미지 생성
  
  
  
  - 얼굴 인식 라이브러리 Mediapipe 의 face_mesh 활용하여 얼굴의 468개의 랜드마크 생성
  - 3D 그래픽 블렌더 조작 라이브러리 bpy 활용하여 468개의 랜드마크 기반으로 안면 3D 객체 생성
  - 안면 3D 객체와 두상 3D 모델 연결

  ### 3. hair_detect : 헤어 영역 추출
  
  - tenseorflow, Keras 기반 DeepLabV3+ 활용하여 헤어 영역 검출
  - Pythorch 기반 DeepLabV3 활용하여 배경제거 (배경에 대한 노이즈 제거하기 위함)
  - opencv의 threshold, Contour 활용하여 노이즈 제거

  ### 4. hair_similarity : 유사한 헤어 매칭
  
  - opencv의 이미지 처리, bitwise 연산, MSE(평균제곱오차) 활용하여 헤어영역의 유사도 측정

  ### 5. 합치기 및 출력

  - create_face 통해 생성된 모델에 맞게 헤어 모델 사이즈 및 좌표 조정
  - THREE.js 활용하여 웹 클라이언트에 3D 객체 출력
  
  
  ## 어려웠던점
   - bpy라이브러리를 사용한 create_face 모듈이 Django 내부에서 정상작동X
   - #### 소켓 프로그래밍 활용하여 문제 해결
  
## <개발환경>
- IDE : PyCharm
- Hardware : AMD Ryzen 7 5700x (3.4GHz, 8 cores), Nvidia GeForce RTX 3060ti, RAM 32GB

## <개발기간>
- 2022년 학기중 (6개월)

## <참조>
