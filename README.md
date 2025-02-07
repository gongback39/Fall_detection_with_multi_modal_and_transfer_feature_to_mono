# Fall_detection_with_multi_modal_and_transfer_feature_to_mono

## 목차

* [모델의 목적](#모델의-목적)
* [사용 데이터셋](#사용-데이터셋)
* [학습모델](#학습모델)
  * [비전모델](#비전모델)
  * [센서모델](#센서모델)
  * [멀티모달모델](#멀티모달모델)
* [학습결과분석](#학습결과분석)
* [멀티모달 to 모노모달](#멀티모달-to-모노모달)
  * [학습결과](#학습결과)
* [코드 사용 설명서](#코드-사용-설명서)
  * [vision](#vision)
  * [sensor](#sensor)
  * [multi_modal](#multi_modal)
 
## 모델의 목적
낙상 감지는 사람의 낙상을 자동으로 감지하여 사고를 신속히 인지하고 도움을 제공하기
위해 설계된 기술 또는 시스템이다. 이는 고령자, 환자 또는 위험 환경에서 작업하는
사람들의 안전을 위해 중요한 연구 및 개발 분야이다.

해당 프로젝트에선 제공된 센서 데이터와 비디오 데이터를 가지고 멀티 모달 모델을 통해
기존에 제시된 성능표에 비해 높은 성능을 보이는 것을 목표로 한다. 

또한 지식증류를 통한 모노 모델을 제안한다. 고령자, 환자, 위험 환경에서 작업하는 사람들
모두 온몸에 센서를 부착할 수 는 없다. 그러므로 비디오 데이터만을 가지고 높은 성능을
보이는 것이 가장 이상적인 환경일 것이다. 그리고 멀티 모달 모델은 기본적으로 높은
용량을 지닌다. 그러므로 cctv 와 같은 소형 기기에 멀티 모달을 탑재하는 것은 여러 제약이
생길 수 있다. 그러므로 지식 증류를 통한 모노 모델 학습이 가능한지와 모델의 크기를
확인해 보겠다

## 사용 데이터셋
[데이터 출처 바로가기](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71641)

이번 모델 학습에 사용할 데이터셋은 AI hub 에서 제공하는 낙상사고 위험동작 영상-센서
쌍 데이터이다. 해당 데이터 셋은 낙상이 포함되거나 포함되지 않은 10 초 간의 데이터로
이루어져 있다. 영상은 60fps 로 찍었으며, 센서 데이터의 time column 은 이에 맞춰
작성되었다. 

데이터 셋 구조를 자세히 살펴보면, 센서 데이터의
경우 12 개의 부위(머리(1), 골반(1), 어깨(2), 상완(2),
좌완(2), 허벅지(2), 하퇴부(2))에 3 종류의 센서,
가속도, 각속도, 자기장 센서가 존재한다. 그리고 각
센서는 x,y,z 축을 기준으로 측정한다. 결과적으로
시간을 뜻하는 행 600 개와 각 센서의 축을 뜻하는
열 108 개로 구성되어 있다. 

반면, 동영상/이미지 데이터의 경우 10 초의 영상이
10 개의 이미지로 저장되어 있다. 10 개의 이미지는
초당 60 개의 이미지 중 대표되는 이미지 하나이다. 또한 각 장면은 8 개의 각도에서 따로
촬영하여, 하나의 장면에 8 개의 동영상/이미지 데이터가 존재한다

낙상 데이터는 전체 1704 개이고 비낙상 데이터는 568 개이다.

## 학습모델
### 비전모델

비전 모델로는 R(2+1)D Net 을 활용하였다. R(2+1)D Net 은 동작인식과 같은 비디오
데이터 분석을 위해 설계된 심층 신경망으로, 3D CNN 을 2D 합성곱과 1D 합성곱으로 나눠
처리한다. 이때 2D 합성곱은 공간의 특징을 1D 합성곱은 시간의 특징을 추출한다.
결론적으로 R(2+1)D Net 은 C3D 와 비교해 계산 복잡도를 줄일 수 있다.

![image](https://github.com/user-attachments/assets/9e467f4d-59ee-45db-b6d5-ff2a50bcaf80)

데이터셋 구성은 간단하다. 먼저 이미지를 160x160 으로 재구성하였다. 160 인 이유는
컴퓨터 GPU 메모리 크기에 맞췄기 때문이다. 카메라 각도로 인한 각 장면의 차이는
고려하지 않았으며, 모두 독립적인 장면이라 고정하고 진행하였다. 구축한 데이터 로더의
형태는 다음과 같다

![image](https://github.com/user-attachments/assets/b14af023-9feb-4787-8f30-908f2f2c20cf)

160x160 크기의 이미지가 10 개 즉 10 초만큼 3 개의 채널로 존재하는 것을 확인할 수 있다.
8 은 배치 크기이다.

![image](https://github.com/user-attachments/assets/192d24b6-17eb-4aa8-9bc9-a0002a9e6074)
![image](https://github.com/user-attachments/assets/dcd7fb01-3b3a-46a2-b463-25cdc9daf81e)

구축한 모델 구조는 다음과 같다. R(2+1)D block 을 1x3x3 3 차원합성곱 연산과 3x1x1
3 차원합성곱 연산으로 구성하였다. 각 합성곱 연산 이후에는 배치 정규화를 통한 내부
공변량 변화 완화와 ReLU 를 통한 비선형성과 기울기 소실 문제 완화를 기대한다. 이와 같이
구성된 R(2+1)Net block 을 4 번 통과하여 비디오 데이터의 특징을 추출한다. 특징 벡터는
평탄화를 거친 후 FC 층에 입력되어 낙상 유무를 판별하게 된다. 출력된 결과값은
5
BCEWithLogitsLoss 를 사용하여 back propagation 을 통해 낙상 유무를 학습한다. 최적화
도구는 Adam 을 사용하였으며, 학습률은 0.0001 로 지정하였다.

### 센서모델
센서 데이터를 학습시키기 위해 1D-CNN 모델을 활용하였다. 1D-CNN 은 1 차원 데이터를
1 차원 kernel 을 통해 합성곱 연산을 수행함을 통해 데이터의 시계열적 특징을 얻을 수
있다. 우리가 가진 센서 데이터는 x,y,z 축을 기준으로 측정한 3 종류의 센서와 12 개의
부위로 구성되어 있으므로, 총 108 개의 1 차원 데이터가 존재하는 것을 알 수 있다. 이때,
개별 부위 사이에 시간적 특징은 적을 것으로 예상되며, 각 부위의 시간적 특징을
결합하고자 한다. 우리는 12 개의 부위를 따로 함성곱 연산을 수행한 뒤, 각 부위의 특징
벡터를 연결하여 낙상을 판별하였다.

위와 같이 구성한 모델 구조에 맞도록 데이터를 전처리 하였다. 먼저 이미지 데이터에 맞춰
센서 데이터를 구성했으므로 한 장면에 8 개의 파일이 존재했다. 하지만 센서 데이터의 경우
어느 각도에서 찍더라도 같은 값을 가지고 있기 때문에 중복되는 데이터를 삭제하였다.
또한 향후 비전 모델과 결합할 것을 대비하여 초당 평균값을 구하여 각 column 당 (1*10)
크기로 구성되도록 수정하였다. 구조를 출력해보면 다음과 같이 나온다.

![image](https://github.com/user-attachments/assets/54f77a80-31f6-402d-ba3a-5103e173e0e2)

 개의 센서종류와 축을 뜻하는 9 와 해당 축 안의 데이터 10 개가 존재하는 것을 확인할 수
있고 이것이 부위 12 개로 묶여 있는 것을 확인할 수 있다. 32 는 배치 크기이다.

모델 구조를 확인하면 다음과 같다.

![image](https://github.com/user-attachments/assets/a38da453-aa95-4cfc-8962-c2242269835d)

입력으로 어떤 한 부위의 가속도, 각속도, 자기장 센서의 x,y,z 값이 들어간다. 입력된
데이터는 크기가 3 인 커널과 합성곱 연산하여 시계열적 특징을 학습시키게 된다. 합성곱
연산을 진행한 각 노드의 결과는 평균값 계산을 통해 하나의 데이터로 표현되고 그림에
보이는 것과 같이 16 개의 데이터로 표현된다. 우린 총 12 개의 부위가 존재하므로 각
부위의 합성곱 연산의 결과를 연결하면 크기가 192(16*12)인 특징 벡터를 구성할 수 있다. 

![image](https://github.com/user-attachments/assets/e2f46025-29ef-4f46-b9ed-227034081038)

구성한 특징 벡터는 FC 층을 통과해
최종적으로 낙상 유무를 표현하게 된다.
출력된 결과는 BCE loss 를 이용해 back
propagation 을 진행하여, 학습을 낙상
유무를 학습한다.

![image](https://github.com/user-attachments/assets/ef17afa6-d9e5-4a2f-8e8c-92fb059fab80)
![image](https://github.com/user-attachments/assets/bb379c69-0b49-44d0-9595-d96ca232d87b)

위에서 설명한 모델을 성능 최적화를 진행해본 결과 위와 같이 모델이 구성되었다. 성능
최적화시 초기 학습율을 0.001 로 지정하였는데, 적당한 학습 속도가 나오다고 생각되어
수정하지 않았다. 최적화 알고리즘은 Adam 을 사용하였다. Max pooling 의 경우 데이터
셋을 확인한 결과 10 초내 낙상 발생 시간이 다른 것을 확인하고 Max pooling 을 통해 높은
특징값을 뽑음으로써 다양한 시간대에서 낙상을 감지할 수 있길 기대하였다. 반면, Max
pooling 시 데이터의 크기가 줄어들어 한 합성곱 연산만에 데이터 사라지는 것을
방지하고자 합성곱 연산의 padding 을 2 로 늘리고 stride 를 1 로 조정하였다. 모든 부위의
특징값을 연결하면 크기가 192 가 되는데 이후 멀티 모달 모델에서 영상 모델과 크기를
맞추기 위해 크기 3, stride 3 의 max pooling 을 통해 특징값의 크기를 64 로 수정하였다.

### 멀티모달모델

**데이터로더 구성**

데이터 로더는 비전모델과 센서모델의 데이터 로더 구조를 거의 똑같이 가져왔다.
다만, 비전 데이터의 경우 장면당 8 개의 각도로 나뉘어 있는 반면, 센서 모델은
전처리할 때 한 장면만 놔두고 지웠기 때문에 데이터의 개수가 맞지 않다. 그러므로
센서 데이터의 경우 중복되더라도 모든 각도를 포함시켰다. 데이터 로더의 형태는
다음과 같다. 8 은 배치 크기이다.

![image](https://github.com/user-attachments/assets/b5749c25-2466-42bd-889d-1719ce66e479)

**모델 구조**

![image](https://github.com/user-attachments/assets/cc6af776-388f-414d-a2b4-5dc45a258348)

멀티 모달 모델도 간단하게 구성하였다. 사전학습한 R(2+1)D 비전 모델과 1D CNN
센서 모델을 통해 각각 크기 64 의 특징 벡터를 얻는다. 얻은 특징 벡터는 결하여
128 크기의 벡터로 변형하고 FC 층을 통해 낙상을 판별한다. 출력된 결과는
BCEWithLogitsLoss 통해 오차를 계산하고 역전파시켜 낙상 유무를 학습한다.
성능 최적화를 진행한 결과 학습률 0.0001 로 Adam 을 사용했을 때 성능이 좋았으며 fc 는
노드 64 개의 층 하나와 ReLU 함수 그리고 노드 1 개의 층을 순서대로 사용했을 때 성능이
좋았다.

## 학습결과분석

다음 표는 사용한 데이터셋의 출처에서 제공한 성능표다

![image](https://github.com/user-attachments/assets/aaf4387e-2b79-445b-93b8-6b90d791434d)

해당 제공처에서는 앙상블 모델을 사용하였으며 정확도 96.5%를 보임을 알 수 있다. 특히,
낙상 판별에 가장 치명적이라고 볼 수 있는 비낙상 precision 이 97%으로 준수한 성능을
보이고 있음을 알 수 있다.

비낙상 precision 은 비낙상이라고 판별한 것 중 실제로 비낙상인 비율로 값이 낮으면
비낙상을 낙상이라고 판별한 비율이 높음을 알 수 있다. 즉, 낙상을 판별하는데 있어 가장
치명적이라고 볼 수 있다. 제시된 성능에 비해 전체적인 성능이 올라가더라도 비낙상
precision 성능은 떨어지지 않는 것이 좋다. 또한 현재 validation 데이터의 비율이 3:1
불균형하므로, f1-score 를 확인할 필요성이 있다. 해당 분석에서는 정확도와 F1-score
그리고 비낙상 precision 을 중점적으로 확인하겠다.

![image](https://github.com/user-attachments/assets/d911aa0c-b94b-40b5-b45e-a09271e6aeb5)

비전 모델의 경우 이전에 제시된 성능과 비슷해 보이지만 조금 더 준수한 성능을 보이고
있음을 알 수 있다. 정확도는 2% 높으며, 비낙상 recall 에 대한 성능이 높아졌음을 알 수
있다.

센서 모델의 경우 제시된 성능과 비슷한 성능을 보인다. 정확도는 약 1% 높으며 비낙상
recall 성능이 많이 올라왔음을 알 수 있다. 하지만 해당 모델을 실제로 사용하기에는 비낙상
precision 이 이전에 제시된 성능보다 많이 하락했음을 알 수 있다. 

멀티 모달 모델의 경우 제시한 모든 모델 중에 가장 높은 성능을 보였다. 전체적으로 이전에
제시된 성능에 비해 높은 성능을 보였으며 비낙상 precision 의 성능이 1%정도 상승한 것을
확인할 수 있다.

## 멀티모달 to 모노모달

앞서 이야기했듯 모델 경량화와 편의성 그리고 자원 한정의 문제를 해결할 수 있는
방법으로 지식 증류를 통한 멀티모달에서 모노모달로의 전이학습을 제안한다. 모노모달의
구조는 비전 모델과 유사하다.

![image](https://github.com/user-attachments/assets/a938678b-ec03-4049-b64c-ffebcc8b268a)

모노모달의 구조는 위 그림의 가장 아래쪽과 같다. 첫 번째 R(2+1)D block 은 멀티 모달의
R(2+1)D Net 의 구조에서 두 번째 R(2+1)D block 의 결과값과의 mse loss 를 계산한다. 두
번째 R(2+1)D block 은 멀티 모달의 R(2+1)D Net 의 구조에서 네 번째 R(2+1)D block 의
결과값과의 mse loss 를 계산한다. 두 loss 값을 평균내어 최종 loss 값을 만든다. 

FC 레이어는 두 결과값의 KLD loss 를 계산한다. 다만 KD loss 계산하기 위해선 확률 분포가
10
필요하다. 현재 출력값이 하나이므로 이를 2 개의 클래스에 대한 값으로 변경해준 후 KD
loss 를 계산한다. 이를 통해 모노 모델의 각 layer 를 멀티모달의 layer 와 값이 비슷하게,
분포를 비슷하게 학습시킬 수 있다.

모노모달의 낙상 판정을 위한 BCE loss 를 포함해 최종적인 loss function 은 다음과 같다

![image](https://github.com/user-attachments/assets/410d85b5-fbeb-45b4-8479-c67d35fa0c8f)

### 학습결과

학습한 멀티모달모델과 지식증류한 모노모달의 크기를 비교해본 결과, 5.2MB 인 멀티 모달 모델의 비해 1.7MB 로 모델 크기가 줄었음을
알 수 있었다.

## 코드 사용 설명서

* 각 모델의 데이터셋과 모델은 모듈화 되어 있어 실행할 파일과 같은 위치에 두고
사용하면 된다.(지식 증류 제외)

* 데이터 폴더는 하위에 N/N 폴더와 Y/()폴더를 가지고 있어야하며, 각 폴더 안엔 각
장면, 각도에 해당하는 폴더와 그 안에 데이터가 들어있어야 한다.

* 저자가 학습을 진행한 코드 파일은 다음과 같다.
  * 비전 모델: vision/fact_CNN_train.ipynb
  * 센서 모델: sensor/1DCNN.ipynb
  * 멀티 모델: Multi_modal/model_train.ipynb
  * 지식 증류 모노 모델: to_mono_modal/model.ipynb
* 아랫문단의 제목은 는 각 상위폴더를 의미한다.

### vision

**데이터 셋 - dataset_for_fact_CNN.py**

```
from dataset_for_fact_CNN import create_dataloaders
val_dataloader = create_dataloaders(data_dir, test_ratio = 0, batch_size=8, image_size = 160,workers=4)
```

파라미터

* Dir: 데이터 위치
* Batch_size: 배치 크기
* Test_ratio : train_test 분리시 해당 비율, 분리하지 않을 시 0
* Image_size : 이미지의 가로 세로 크기 (정수)
* Workers : 데이터로더 생성시 사용하는 nume_workers 파라미터의 값

**모델 - vison_model.py**

```
from vision_model import R2Plus1DNet
model = R2Plus1DNet(num_classes=1).to(DEVICE)
```

**모델 학습, validation – vison_model.py**

```from vision_model import val
from vision_model import train
avg_val_loss, val_acc, cm = val(model, val_dataloader, criterion, DEVICE)
avg_loss, acc = train(model, train_dataloader, val_dataloader, optimizer, criterion, device=DEVICE, epoches=EPOCHS)
```

Val

- 반환값
  - 로스값, 정확도, confusion matrix

- 파라미터
  - Model: 모델
  - Val_loader: 데이터 로더
  - Criterion: 오차함수
  - Device: PyTorch의 device

Train

- 반환값
  - Validation data의 로스값, 정확도

- 파라미터
  - Model: 모델
  - Train_loader: 학습 데이터 로더
  - Val_loader: 검증 데이터 로더
  - Optimizer: 최적화 도구
  - Criterion: 오차함수
  - Device: PyTorch의 device
  - Epochs: 에포크 횟수


**학습한 모델 - ~.pth**

```
checkpoint = torch.load("24_12_13.pth", map_location=DEVICE)
model = R2Plus1DNet(num_classes=1).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
```

### sensor

**데이터 셋- sensordataset.py**

해당 파일에서 중복되는 센서데이터를 지우진 않는다. 중복되는 센서데이터는 지운 후 sensor_training, sensor_validation 폴더에 저장 됨.

```
from sensordataset import SensorDataset
data_types = ['Segment Acceleration', 'Segment Angular Velocity', 'Sensor Magnetic Field']
train_dataset = SensorDataset(input_dir='./sensor_Training', data_types=data_types)
```

파라미터

- Input_dir: 데이터 위치
- Data_types: 3 종류의 센서 중 사용하고 싶은 센서 (리스트)
  - Segmant Acceleration
  - Segment Angular Velocity
  - Sensor Magnetic Field

**모델 - sensor_model.py**

```
from sensor_model import FallDetection1DCNN
sensor_model = FallDetection1DCNN(num_classes=1).to(DEVICE)
```

**학습한 모델 - ~.pth**

```
sensor_checkpoint = torch.load("sensor_24_12_15.pth", map_location=DEVICE)
sensor_model = FallDetection1DCNN(num_classes=1).to(DEVICE)
sensor_model.load_state_dict(sensor_checkpoint, strict=False)
```

### multi_modal

**데이터 셋 - dataset.py**

```
from dataset import create_dataloaders
train_loader = create_dataloaders("../data/Training/01.원천데이터", batch_size=8, test_ratio =
0, image_size=160, workers=16)
```

파라미터

- Dir: 데이터 위치
- Batch_size: 배치 크기
- Test_ratio: train_test 분리 시 해당 비율, 분리하지 않을 시 0
- Image_size: 이미지의 가로 세로 크기 (정수)
- Workers: 데이터로더 생성 시 사용하는 num_workers 파라미터의 값


**모델 - vison_model.py ,sensor_model.py**

비전폴더, 센서폴더의 모델과 같지만 feature extraction 층까지만 존재

**모델 - multi_modal.py**

vision_24_12_13.pth 와 sensor_24_12_15.pth 를 지니고 있어야 함

```
from multi_modal import MultiModalNet
teacher_model= MultiModalNet(num_classes=1).to(DEVICE)
```

**학습한 모델 - multi_modal_24_12_15.pth**

```
from multi_modal import MultiModalNet
multi_modal_checkpoint = torch.load("multi_modal_24_12_15.pth", map_location=DEVICE)
teacher_model= MultiModalNet(num_classes=1).to(DEVICE)
teacher_model.load_state_dict(multi_modal_checkpoint, strict=False)
```





