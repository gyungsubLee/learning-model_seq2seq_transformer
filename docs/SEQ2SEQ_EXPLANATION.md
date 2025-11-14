# Seq2Seq with Attention 기계번역 모델 설명

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [모델 아키텍처](#모델-아키텍처)
3. [코드 구조](#코드-구조)
4. [상세 구현 설명](#상세-구현-설명)
5. [학습 과정](#학습-과정)
6. [평가 및 추론](#평가-및-추론)

---

## 프로젝트 개요

이 프로젝트는 **Sequence-to-Sequence (Seq2Seq) 모델**과 **Attention 메커니즘**을 사용하여 기계번역을 수행하는 시스템입니다.

### 주요 특징

- **Encoder-Decoder 아키텍처**: GRU 기반 인코더와 디코더
- **Bahdanau Attention**: 디코더가 인코더의 모든 hidden state에 접근 가능
- **Teacher Forcing**: 학습 시 실제 정답을 다음 입력으로 사용하여 학습 안정화
- **지원 언어**: 영어 ↔ 프랑스어, 영어 ↔ 한국어

---

## 모델 아키텍처

### 1. 전체 구조

```
Input Sentence → Encoder → Hidden States → Attention → Decoder → Output Sentence
```

### 2. Encoder (EncoderRNN)

```
단어 시퀀스 → Embedding → Dropout → GRU → Hidden States
```

**역할**: 입력 문장을 고정된 크기의 벡터 표현(hidden state)으로 변환

**구성요소**:

- **Embedding Layer**: 단어 인덱스를 밀집 벡터로 변환
- **GRU Layer**: 순차적 정보를 처리하고 문맥을 인코딩
- **Dropout**: 과적합 방지

### 3. Attention Mechanism (BahdanauAttention)

**역할**: 디코더가 출력 생성 시 입력 문장의 어느 부분에 집중할지 결정

**수식**:

```
score = Va(tanh(Wa(query) + Ua(keys)))
attention_weights = softmax(score)
context = Σ(attention_weights * encoder_outputs)
```

**핵심 아이디어**:

- 디코더의 현재 hidden state(query)와 인코더의 모든 출력(keys)을 비교
- 가장 관련성 높은 인코더 출력에 높은 가중치 부여
- 가중합으로 context vector 생성

### 4. Decoder with Attention (AttnDecoderRNN)

```
Start Token → Embedding → [Embedding + Context] → GRU → Linear → Output Word
                ↑                                      ↓
                └──────────── Attention ←──────────────┘
```

**역할**: 인코더의 정보와 attention을 활용하여 번역된 문장 생성

**구성요소**:

- **Embedding Layer**: 이전 출력 단어를 벡터로 변환
- **Attention Module**: 현재 시점에서 중요한 입력 부분 계산
- **GRU Layer**: 임베딩과 context를 결합하여 처리
- **Output Layer**: 다음 단어의 확률 분포 생성

---

## 코드 구조

### 파일 구성

```
.
├── main_seq2seq.py          # 메인 실행 파일
├── seq2seq_model.py         # 모델 정의 (Encoder, Decoder, Attention)
├── language_processor.py    # 텍스트 전처리 및 데이터 준비
├── utils.py                 # 시각화 및 유틸리티 함수
└── data/
    ├── eng-fra.txt         # 영어-프랑스어 병렬 코퍼스
    └── eng-kor.txt         # 영어-한국어 병렬 코퍼스
```

---

## 상세 구현 설명

### 1. language_processor.py

#### Lang 클래스

```python
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}      # 단어 → 인덱스 매핑
        self.word2count = {}      # 단어 빈도수
        self.index2word = {0: "SOS", 1: "EOS"}  # 인덱스 → 단어 매핑
        self.n_words = 2          # 어휘 크기 (SOS, EOS 포함)
```

**기능**:

- 어휘 사전 구축 및 관리
- 단어 ↔ 인덱스 변환
- `SOS_token(0)`: 문장 시작 토큰
- `EOS_token(1)`: 문장 종료 토큰

#### 데이터 전처리 파이프라인

```python
prepareData(lang1, lang2, reverse=False)
```

**단계**:

1. **텍스트 파일 읽기**: 탭으로 구분된 병렬 문장 쌍
2. **정규화**:
   - 유니코드 → ASCII 변환 (프랑스어 악센트 처리)
   - 소문자화
   - 구두점 정리
3. **필터링**:
   - 최대 길이 10단어 이하
   - 특정 패턴으로 시작하는 문장만 선택
4. **어휘 구축**: 모든 단어를 Lang 객체에 추가

### 2. seq2seq_model.py

#### EncoderRNN

```python
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
```

**Forward Pass**:

1. 입력 인덱스 → 임베딩 벡터
2. Dropout 적용
3. GRU를 통해 순차 처리
4. 모든 hidden state와 마지막 hidden state 반환

#### BahdanauAttention

```python
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        self.Wa = nn.Linear(hidden_size, hidden_size)  # query 변환
        self.Ua = nn.Linear(hidden_size, hidden_size)  # key 변환
        self.Va = nn.Linear(hidden_size, 1)            # score 계산

    def forward(self, query, keys):
        # 덧셈 attention 메커니즘
        # ( 일반적으로 '내적'을 많이 사용하지만, 다른 케이스를 학습을 위해 덧셈 방식 사용 )
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights
```

**Attention 계산 과정**:

1. **Query**: 디코더의 현재 hidden state
2. **Keys**: 인코더의 모든 출력
3. **Score 계산**: `Va(tanh(Wa(query) + Ua(keys)))`
4. **Softmax**: score를 확률로 변환
5. **Context Vector**: 가중합 계산

#### AttnDecoderRNN

```python
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
```

**Forward Step**:

1. 현재 입력 단어 임베딩
2. Attention으로 context vector 계산
3. 임베딩과 context 결합 → GRU 입력
4. GRU 출력 → Linear layer → 단어 확률 분포
5. **Teacher Forcing**: 학습 시 실제 정답을 다음 입력으로 사용

---

## 학습 과정

### 1. 데이터 준비 (main_seq2seq.py)

```python
def get_dataloader(batch_size, target_lang='fra'):
    # 1. 데이터 로드 및 전처리
    input_lang, output_lang, pairs = prepareData('eng', target_lang, True)

    # 2. 문장을 인덱스 시퀀스로 변환
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    # 3. 각 문장을 인덱스로 변환하고 EOS 토큰 추가
    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)

    # 4. DataLoader 생성
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=batch_size)
```

### 2. 학습 루프 (train_epoch)

```python
def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion):
    for data in dataloader:
        input_tensor, target_tensor = data

        # 1. 그래디언트 초기화
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # 2. Forward Pass
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden,
                                       target_tensor)  # Teacher Forcing

        # 3. Loss 계산 (NLLLoss)
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )

        # 4. Backward Pass
        loss.backward()

        # 5. 파라미터 업데이트
        encoder_optimizer.step()
        decoder_optimizer.step()
```

### 3. 하이퍼파라미터

```python
hidden_size = 128       # GRU hidden state 크기
batch_size = 32         # 배치 크기
n_epochs = 200          # 학습 에포크
learning_rate = 0.001   # 학습률
dropout = 0.1           # Dropout 비율
```

### 4. Loss Function

- **NLLLoss** (Negative Log-Likelihood Loss)
- `ignore_index=SOS_token`: SOS 토큰은 loss 계산에서 제외
- Log-Softmax 출력과 함께 사용

---

## 평가 및 추론

### 1. 평가 모드 (evaluate)

```python
def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():  # 그래디언트 계산 비활성화
        # 1. 입력 문장을 텐서로 변환
        input_tensor = tensorFromSentence(input_lang, sentence)

        # 2. 인코더 실행
        encoder_outputs, encoder_hidden = encoder(input_tensor)

        # 3. 디코더 실행 (Teacher Forcing 없음)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(
            encoder_outputs, encoder_hidden
        )

        # 4. 가장 높은 확률의 단어 선택
        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        # 5. 인덱스를 단어로 변환
        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                break
            decoded_words.append(output_lang.index2word[idx.item()])
```

**Teacher Forcing vs Inference**:

- **학습 시 (Teacher Forcing)**: 실제 정답을 다음 입력으로 사용

  - 장점: 빠른 학습, 안정적 수렴
  - 단점: 실제 추론과 불일치

- **추론 시**: 모델의 이전 출력을 다음 입력으로 사용
  - 장점: 실제 사용 환경과 동일
  - 단점: 에러 누적 가능성

### 2. 무작위 평가 (evaluateRandomly)

```python
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])  # 입력 (한국어/프랑스어)
        print('=', pair[1])  # 정답 (영어)
        output_words, _ = evaluate(encoder, decoder, pair[0],
                                   input_lang, output_lang)
        print('<', ' '.join(output_words))  # 모델 출력
```

---

## 실행 예제

### 메인 실행 흐름

```python
if __name__ == '__main__':
    # 1. 하이퍼파라미터 설정
    hidden_size = 128
    batch_size = 32
    n_epochs = 200
    target_lang = 'kor'  # 또는 'fra'

    # 2. 데이터 로드
    input_lang, output_lang, train_dataloader, pairs = get_dataloader(
        batch_size, target_lang=target_lang
    )

    # 3. 모델 초기화
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

    # 4. 학습
    train(train_dataloader, encoder, decoder, n_epochs,
          print_every=5, plot_every=5)

    # 5. 평가 모드로 전환
    encoder.eval()
    decoder.eval()

    # 6. 결과 확인
    evaluateRandomly(encoder, decoder)
```

---

## 주요 기법 정리

### 1. Attention의 역할

- **문제**: 긴 문장에서 인코더의 마지막 hidden state만으로는 정보 손실
- **해결**: 모든 인코더 출력에 접근하여 필요한 정보 선택적으로 사용
- **효과**: 긴 문장 번역 성능 향상, 정렬 정보 시각화 가능

### 2. Teacher Forcing

- **개념**: 학습 시 모델의 이전 출력 대신 실제 정답을 다음 입력으로 사용
- **장점**: 학습 속도 향상, 안정적 수렴
- **단점**: 추론 시 불일치 (Exposure Bias)

### 3. Padding과 Masking

- **Padding**: 가변 길이 문장을 고정 길이로 맞춤 (MAX_LENGTH=10)
- **NLLLoss ignore_index**: 패딩 토큰을 loss 계산에서 제외

### 4. 최적화 기법

- **Adam Optimizer**: 적응적 학습률 조정
- **Dropout (0.1)**: 과적합 방지
- **Gradient Clipping**: 기울기 폭발 방지 (코드에는 미구현)

---

## 모델 성능 개선 방향

1. **Bidirectional Encoder**: 양방향 GRU로 문맥 정보 향상
2. **Beam Search**: Greedy 대신 여러 후보 고려
3. **Layer Normalization**: 학습 안정화
4. **Scheduled Sampling**: Teacher Forcing 비율 점진적 감소
5. **Subword Tokenization**: BPE/WordPiece로 OOV 문제 해결
6. **Transformer 모델**: 병렬 처리 및 장거리 의존성 개선

---

## 참고 자료

- Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2014)
- Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder" (2014)
- PyTorch Seq2Seq Tutorial: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
