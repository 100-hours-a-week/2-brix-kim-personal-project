# Feature Map 기반 CNN 모델 최적화: Rice Image Dataset 사례 연구

## 프로젝트 소개
&nbsp;&nbsp;최근 다양한 CNN(Convolutional Neural Network) 기반의 이미지 분류 모델들이 제안되며 이미지 인식 분야에서 뛰어난 성능을 보이고 있다. 하지만 이러한 모델들은 구조의 복잡성이나 파라미터 수에 따라 연산 비용과 메모리 사용량에 큰 차이를 보이며, 특히 데이터셋의 특성에 따라 성능 및 효율성이 달라질 수 있다.<br/>

&nbsp;&nbsp;실제 응용 환경에서는 모델의 경량화 또한 중요한 과제로 떠오르고 있다. 모바일 기기, 임베디드 시스템, 자동화 기계 등에서는 모델의 정확도뿐만 아니라 처리 속도와 자원 효율성이 중요한 요소로 작용하기 때문이다.<br/>

&nbsp;&nbsp;따라서 본 프로젝트에서는 ‘Rice Image Dataset’을 활용하여 여러 CNN 기반 모델들이 해당 데이터셋에서 어떤 성능을 보이는지 비교하고 모델별 특징 및 효율성을 분석하고자 한다. 특히, 이미지 분류 과정에서 생성되는 Feature map을 시각화함으로써 각 모델이 어떤 방식으로 이미지를 인식하고 구분하는지 직관적으로 이해하고자 하였다.<br/>

&nbsp;&nbsp;이를 통해 더 이상 특징을 제대로 추출하지 못 하는 Layer를 일부 제거하거나 Filter의 수를 줄임 성능저하 없이 예측 효율성을 개선할 수 있는 가능성도 살펴보고자 한다. 이러한 분석을 통해 단순 정확도 비교를 넘어서 실제 응용에 적합한 효율적인 모델을 선정하기 위한 방법을 살펴보고자 한다.<br/>

## 프로젝트 기간 및 인원
- 프로젝트 기간: 2025.03.19. ~ 2025.03.30.
- 프로젝트 인원: 모델설계 및 분석 1명(본인)

## 사용기술 및 Tool
- Python
- NumPy, Pandas
- Tensorflow (Vanila CNN, GoogLeNet, VGG16)
- Matplotlib

## Final Model 구조
| Layer (type)                         | Output Shape         | Param #     |
|-------------------------------------|----------------------|-------------|
| input_layer_10 (InputLayer)         | (None, 224, 224, 3)  | 0           |
| conv2d_32 (Conv2D)                  | (None, 224, 224, 8)  | 224         |
| max_pooling2d_20 (MaxPooling2D)     | (None, 112, 112, 8)  | 0           |
| conv2d_33 (Conv2D)                  | (None, 112, 112, 16) | 1,168       |
| conv2d_34 (Conv2D)                  | (None, 112, 112, 16) | 2,320       |
| conv2d_35 (Conv2D)                  | (None, 112, 112, 16) | 2,320       |
| max_pooling2d_21 (MaxPooling2D)     | (None, 56, 56, 16)   | 0           |
| global_average_pooling2d_6 (GlobalAveragePooling2D) | (None, 16) | 0     |
| flatten_10 (Flatten)                | (None, 16)           | 0           |
| dense_20 (Dense)                    | (None, 512)          | 8,704       |
| dropout_3 (Dropout)                 | (None, 512)          | 0           |
| dense_21 (Dense)                    | (None, 5)            | 2,565       |

## 학습결과 DB연동
![alt text](/assets/db_1.png)
수업시간에 활용한 aiven에 서버를 두고 MySQL을 이용하여 학습결과를 저장하였다.

- MySQL 설정
    ```python
    conn = mysql.connector.connect(
        user='...',
        password='...',
        host='...',
        port=16027,
        database='defaultdb'
    )

    cursor = conn.cursor()
    ```
- TABLE 생성
    ```python
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS CNN_training_results (
        id INT AUTO_INCREMENT PRIMARY KEY,
        model_name VARCHAR(255),
        epoch INT,
        accuracy FLOAT,
        loss FLOAT,
        val_accuracy FLOAT,
        val_loss FLOAT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    ```
- 학습결과 저장(CNN)
    ```python
    model_name = 'CNN'

    for epoch in range(len(history.history['accuracy'])):
        acc = history.history['accuracy'][epoch]
        loss = history.history['loss'][epoch]
        val_acc = history.history['val_accuracy'][epoch]
        val_loss = history.history['val_loss'][epoch]

        cursor.execute("""
            INSERT INTO CNN_training_results (model_name, epoch, accuracy, loss, val_accuracy, val_loss)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (model_name, epoch + 1, acc, loss, val_acc, val_loss))
    ```
    history에 저장된 학습결과를 TABLE에 입력한다.

- Commit
    ```python
    conn.commit()
    cursor.close()
    conn.close()
    ```

- TABLE 확인
    ```python
    query = "SELECT * FROM CNN_training_results ORDER BY epoch"
    df = pd.read_sql(query, conn)
    conn.close()

    from IPython.display import display
    display(df)
    ```

    | id  | model_name        | epoch | accuracy  | loss     | val_accuracy | val_loss | timestamp           |
    |-----|-------------------|-------|-----------|----------|--------------|----------|----------------------|
    | 1   | CNN               | 1     | 0.955617  | 0.127012 | 0.974267     | 0.074242 | 2025-03-28 01:26:37 |
    | 14  | CNN Light         | 1     | 0.935450  | 0.186255 | 0.944933     | 0.152867 | 2025-03-28 01:34:42 |
    | 41  | GoogLeNet         | 1     | 0.841133  | 0.378455 | 0.967800     | 0.097555 | 2025-03-28 02:01:56 |
    | 70  | GoogLeNet Light   | 1     | 0.798717  | 0.473984 | 0.962067     | 0.106132 | 2025-03-28 02:14:34 |
    | 99  | VGG16             | 1     | 0.949567  | 0.124866 | 0.993200     | 0.023065 | 2025-03-28 02:22:31 |
    | ... | ...               | ...   | ...       | ...      | ...          | ...      | ...                  |
    | 177 | Final Model       | 35    | 0.996600  | 0.011187 | 0.996933     | 0.010730 | 2025-03-28 02:55:58 |
    | 178 | Final Model       | 36    | 0.996883  | 0.010867 | 0.996067     | 0.012451 | 2025-03-28 02:55:58 |
    | 179 | Final Model       | 37    | 0.996933  | 0.010792 | 0.996933     | 0.010512 | 2025-03-28 02:55:59 |
    | 180 | Final Model       | 38    | 0.996933  | 0.009820 | 0.996867     | 0.010371 | 2025-03-28 02:55:59 |
    | 181 | Final Model       | 39    | 0.997017  | 0.010021 | 0.996933     | 0.010352 | 2025-03-28 02:55:59 |

## Project 파일
- [Project 정리](/2025-03-22-cnn_project.md)
- [모델 설계 colab link](https://colab.research.google.com/drive/1WgjlYTLEkDyacbimKDHpdKw0LBZn4aTi?usp=sharing)
- [GoogLeNet 학습불가 colab link](https://colab.research.google.com/drive/1Wt_dMel9Vv8HwzGIlkuvochkIFNWLsv0?usp=sharing)
- [모델 성능평가 colab link](https://colab.research.google.com/drive/1x-QuXOEOWcSqWvZlDuVOeBu88Hb8jMWO?usp=sharing)

## 프로젝트 후기
&nbsp;&nbsp;개인 프로젝트를 통해 원하는 연구를 해볼 수 있었다. Feature Map의 여러 변수들은 어떤 작용을 하는지 알 수 없기 때문에 시각적으로 쓸모가 있는 Filter, Layer인지 알 수 없다. 따라서 시각적으로 Feature Map을 통한 모델 경량화는 시도하지 않는 분야이다.<br/>

&nbsp;&nbsp;하지만 개인 프로젝트로써 과감하게 시도해 볼 가치가 있다고 생각했으며 만족스러운 결과가 나왔다. 또한 DB에 학습결과를 연동해보면서 DB에 대한 공부도 할 수 있는 기회가 되었다.<br/>