# t-SNE_tutorial
simple tutorial for t-SNE

이 repository는 dimensionality reduction에 대해 막 관심이 생긴 초보자들(저같은)을 위해 만들어졌습니다. 여러 dimensionality reduction 기법의 개념을 살펴보고, 이 중 t-SNE를 다른 모듈의 구현체없이 직접 구현해본 후, scikit-learn 모듈을 이용하여 실제 실습해보는 순서로 구성하였습니다. 실습을 진행해볼 데이터셋은 [MNIST](http://yann.lecun.com/exdb/mnist/)입니다. 이론적인 토대는 주로 고려대학교 강필성 교수님의 [유튜브 강의](https://www.youtube.com/watch?v=ytRmxBvyGG0&list=PLetSlH8YjIfWMdw9AuLR5ybkVvGcoG2EW)를 참고하였음을 밝힙니다. 

## 목차
1. [Dimensionality Reduction Overview](#dimensionality-reduction-overview)
2. [Concepts of t-distributed Stochastic Neighbor Embedding(t-SNE)](#concepts-of-t-distributed-stochastic-neighbor-embeddingt-sne)
3. [t-SNE Implementation](#t-sne-implementation)

---

## Dimensionality Reduction Overview
dimensionality reduction(차원축소)은 데이터가 가지고 있는 객체들을 설명하는 요소들의 수를 줄인다는 뜻입니다. 데이터가 가진 객체들을 설명하는 요소라는 것이 어떤 의미일까요? 아래의 표를 봅시다.
| 이름 | 성별 | 나이  | 병명  | 몸무게  | 키   |
| :---: | :--: | :--:  | :--:   | :--: | :--: |
| A |  F  |  23  | Diabetes Meillitus | 60 | 166 |
| B |  M  |  50  | Cystitis  | 97 | 172 |
| C |  M  |  45  | Acute gastric ulcer | 55 | 168 |
| D |  F  |  48  | Cholecystitis | 57 | 175 |
| E |  F  |  87  | Gingivitis | 68 | 158 |
| F |  M  |  36  | Acute pericarditis | 70 | 182 |

위 표에는 환자 6명의 데이터가 이름, 성별, 나이, 병명, 몸무게, 키라는 6개의 요소(feature)를 통해 설명되어 있습니다. 따라서 위 데이터셋에는 6개의 객체가 존재하며, 이를 설명하기 위해 6개의 요소가 존재하므로 dimension(차원)이 6이라고 할 수 있을 것입니다.

이때, dimensionality reduction은 그 차원을 어떤 방식으로든 줄이는 것을 의미합니다. 예를 들어, 위 데이터 셋에서 이름을 삭제하기로 했다고 가정합시다. 그러면 아래와 같이 되겠죠?
| 성별 | 나이  | 병명  | 몸무게  | 키   |
| :--: | :--:  | :--:   | :--: | :--: |
|  F  |  23  | Diabetes Meillitus | 60 | 166 |
|  M  |  50  | Cystitis  | 97 | 172 |
|  M  |  45  | Acute gastric ulcer | 55 | 168 |
|  F  |  48  | Cholecystitis | 57 | 175 |
|  F  |  87  | Gingivitis | 68 | 158 |
|  M  |  36  | Acute pericarditis | 70 | 182 |

이것도 일종의 dimensionality reduction을 수행한 결과라고 할 수 있습니다. 수행한 결과의 dimensionality는 5가 되겠죠. 

표 데이터 뿐만 아니라 이미지 데이터에서도 이러한 dimensionality를 정의할 수 있습니다. MNIST로 예를 들어봅시다. 아래 사진은 MNIST의 이미지 일부를 표시한 결과입니다. MNIST의 숫자 이미지 28 by 28, 즉 784개의 픽셀로 이루어져있습니다. MNIST의 이미지 하나를 데이터 객체라고 생각하면 각 객체는 784개의 요소로 이뤄져있으므로 MNIST의 dimensionality는 784입니다.

<p align="center"><img src="https://user-images.githubusercontent.com/112034941/194852600-9cfb7772-5f22-41cd-8ea8-de48f9e440aa.png" height="350px" width="350px"></p>

그런데, dimensionality reduction이 필요한 이유는 무엇일까요? 앞서 설명한 바에 따르면 dimensionality는 데이터 객체를 설명하기 위한 요소의 수입니다. 그렇다면 자연스럽게 생각할 수 있는것은(*사실 제가 처음 배울때 생각했던 겁니다*) dimensionality reduction을 하면 오히려 안 좋은 것 아닌가 하는 의문입니다. 데이터 객체를 설명하는 요소를 줄이는 일이니까요. 이에 대해 결론부터 말하자면 ***그렇지 않다!*** 는 겁니다. 이에는 크게 두 가지 이유가 있는데, 첫째는 dataset dimension이 감소해도 설명력이 유지될 수 있기 때문이고, 둘째는 dataset의 dimension이 클수록 여러 문제가 발생할 수 있다는 것입니다. 이에 대해 조금 더 자세히 설명하겠습니다.
1. 데이터셋의 intrinsic dimension은 일반적으로 데이터셋이 원래 가지고 있는 dimension에 비해 작습니다. Intrinsic dimension을 쉽게 말하자면, 데이터셋이 전달하고자 하는 정보를 설명하기 위해 꼭 필요한 dimension 요소의 수입니다. 예를 들어, MNIST의 경우 데이터셋이 전달하고자 하는 정보는 *'손으로 쓴 숫자 0~9'* 입니다. 이 정보를 전달하기 위해 784개의 픽셀이 다 필요할까요? 그렇지 않을겁니다. 예를 들어 아래 사진처럼 MNIST의 숫자 1 이미지를 적당히 잘라내도 여전히 1임을 알아 볼 수 있습니다. 물론 실제로 이런 식으로 dimensionality reduction을 하지는 않지만요. 이처럼, 대체로 데이터셋의 dimension을 줄여도 원하는 task에 대한 설명력을 유지할 수 있습니다. 이것이 dimensionality reduction을 하는 첫번째 이유입니다.
<p align="center"><img src="https://user-images.githubusercontent.com/112034941/194868403-9919c2fe-445f-4904-a03e-841e153936f0.png" height="250px" width="450px"></p>

2. dataset의 dimension이 증가하면 일반적으로 여러 문제가 발생합니다. 사실, 첫번째 이유는 사실 dimensionality reduction을 해"도" 좋은 이유지 해"야할"이유는 아닙니다. 여기에 두 번째 이유가 함께 작용하기 때문에 dimensionality reduction이 필요한 것입니다. 이론적으로는 만약 데이터셋의 모든 관측이 정확하게 이뤄져서 올바르게 데이터셋에 기록되었다면 dimension이 늘어나더라도 이를 통해 모델링했을 때의 결과가 나빠질 일은 없습니다. 하지만 현실세계에서는 데이터셋에는 여러 부정확한 관측치가 기록되기도 하고 기록과정에서의 오류도 존재하는 편입니다. 때문에 dimension이 늘어나는 경우 이를 통해 모델링하는 경우 그 성능이 저하되는 경우가 발생합니다. 이에 더해, data의 sparsity 문제, 계산량 증가 또한 dimensionality의 증가에 따른 문제라고 볼 수 있습니다.

결론적으로, dimensionality reduction이 필요한 이유는 줄여도 설명력이 유지되고, 줄이지 않으면 오히려 성능이 떨어지는 등의 문제가 생길 수 있기 때문입니다. 그래서 여러 차원축소기법을 적용할 때는 비슷한 설명력을 유지하면서 dimension을 줄이는 것을 핵심으로 하여 적용해야 합니다.

차원축소 기법에는 다양한 기법이 존재합니다. 그리고 이러한 방법론들을 분류하려고 한다면 가장 쉽게 쓰일 수 있는 기준으로는 방법론들의 모델과의 독립성(supervised vs unsupervised)과 기존 dimension 요소를 선택하는가 아닌가(selection vs extraction)가 있습니다. 
1. 만약 어떤 dimensionality reduction 방법론이 supervised 방법론이라면, 그 방법론은 방법론을 수행하는 과정 중에 모델(통계적이거나 머신러닝적인)이 필요합니다. 예를 들자면 유명한 forward selection 방법론이나 backward elimination 방법론들이 대표적인 이러한 supervised 방법이라고 할 수 있습니다. 반대로 unsupervised 방법론의 경우 모델에 의한 성능 평가 등의 과정이 포함되지 않는 방법론들입니다.
2. 만약 어떤 dimensionality reduction 방법론이 selection 방법론이라면, 그 방법론에 의해 차원축소된 데이터셋의 dimension 요소는 원래 데이터셋의 dimension 요소와 동일합니다. 만약 표 데이터셋에 selection 방법론을 활용하면 결과적으로 차원축소의 결과물은 원래 column들의 부분집합으로 이뤄지게 될 것입니다. 앞서 이미 예시로 들었던 forward selection, backward elimination 방법론들이 대표적인 selection 방법론입니다. 반대로, extraction 방법론의 경우 차원축소의 결과물이 기존 dimension 구성 요소의 연산 결과가 됩니다. 예를 들면 PCA와 같은 알고리즘이 이에 해당하겠죠.

이러한 기준에 따라 유명한 dimensionality reduction 방법론들을 분류하면 아래와 같이 분류할 수 있습니다.

|              | selection                                                                      | extraction                    |
|--------------|--------------------------------------------------------------------------------|-------------------------------|
| supervised   | forward selection, backward elimination, stepwise selection, genetic algorithm | lasso, elastic net regression |
| unsupervised | filter: correlation analysis                                                   | PCA, MDS, ISOMAP, LLE, t-SNE  |

물론 이외에도 KPCA, LDA 등등 다양한 차원축소 기법이 존재합니다. 일단 이 repository에서는 t-SNE 방법론을 살펴보고, 이를 구현해보려고 합니다. 그럼, t-SNE의 개념부터 살펴봅시다.

---

## Concepts of t-distributed Stochastic Neighbor Embedding(t-SNE)
t-SNE의 방법론적 핵심을 요약하면, 데이터셋의 객체들이 그들의 이웃과의 거리 정보를 거리에 따라 감소하는 확률로써 반영하여(즉, 거리가 가까울수록 서로간의 확률값이 높아짐.) 각 객체사이의 확률값이 저차원에서도 보존되도록 차원을 축소시키는 것입니다. 쓰면서도 머리가 어지럽네요. 이게 무슨 뜻인지 차근차근 알아봅시다.

### Stochastic Neighbor Embedding(SNE)
t-SNE는 SNE로부터 출발한 방법입니다. SNE는 두 데이터 이웃 데이터 객체간의 거리를 stochastic하게 정의하는 것이 핵심적인 아이디어입니다. SNE에서는 이 아이디어를 각 원래 데이터셋의 차원에서 객체 $i$와 $j$의 euclidean distance를 일종의 조건부 확률로 전환하여 이를 두 객체의 유사도로 함으로써 구현하였습니다. SNE에서 $i$ 기준 $j$와의 유사도를 $p_{j|i}$라 하고, 축소된 차원에의 $i$ 기준 $j$의 유사도를 $q_{j|i}$라 하면 아래와 같습니다.
<p align="center"><img src="https://user-images.githubusercontent.com/112034941/195049568-a448467a-2bb3-4f5a-8d00-e8ae42a7cb30.png" height="200px" width="600px"></p>

단, $p_{i|i}, q_{i|i} = 0$입니다. 이유는 이 방법론이 데이터의 전체적 분포를 보기 위해 객체의 이웃과의 관계에 집중하는 것을 목적으로 하기 때문입니다. 굳이 자기 자신에 대한 확률을 고려하여 다른 이웃과의 관계를 희석시킬 필요가 없는 셈이죠.

SNE에서는 $p_{j|i}$의 전체적 분포와 $q_{j|i}$의 전체적 분포를 유사하게 하게 하는 $y_i$를 찾아내어 축소된 공간에서도 원래 공간에서의 거리가 가까울수록 높게 보존하는 방식의 embedding을 구현합니다. 여기까지 이해가 되셨다면, 자연스럽게 $p_{j|i}$의 전체적 분포와 $q_{j|i}$의 전체적 분포를 비슷하게 하는 $y_i$를 찾아내기 위해 두 확률분포의 차이를 어떤 식으로 측정할 것인지 궁금하실겁니다. SNE(t-SNE에서도)에서는 이를 두 확률분포의 KL(Kullback-Leiber) divergence를 cost function으로 하는 최적화문제를 해결하는 방식으로 접근하였습니다. 원래 공간에서의 객체 $x_i$의 확률분포를 $P_{i}$라 하고 객체 $y_i$의 확률분포를 $Q_{i}$라 하면 cost function은 아래와 같습니다. 
<p align="center"><img src="https://user-images.githubusercontent.com/112034941/195084273-55a1a174-12f7-4eff-80c9-bc9fc90a37b6.png" height="100px" width="600px"></p>

위 cost function을 계산하는데 필요한 요소를 자세히 봅시다. 결국 $p_{j|i}$와 $q_{j|i}$를 계산하면 되고, 각각을 계산하기 위해서는 $x_i$와 $y_i$, 그리고 자세히 보시면  $p_{j|i}$에 있는 $\sigma_i$를 알아야 합니다. $x_i$는 데이터셋에서 주어지는 값입니다. 그리고 $y_i$는 우리가 최적화 문제를 통해 찾아야 하는 값이죠. 그러면 $\sigma_i$가 대체 뭘까요? 이를 알기 위해서는 우선 perplexity부터 알아야 합니다.
> perplexity
> 
> perplexity는 $2^{entropy}$ 이며, 어떤 확률분포를 $p(x)$라 할 때 $p(x)$의 entropy는 $\sum_x -p(x)log_2(p(x))$입니다. 식을 보시면 아시겠지만, 어떤 확률분포가 고르게 분포하면 분포할수록 해당 확률분포의 entropy가 높아집니다. 이를 SNE에 적용해보면, 데이터 객체 $x_i$에 대한 확률분포 $P_i$의 entropy는 $\sum_j -p_{j|i}log_2(p_{j|i})$라 할 수 있고, 결국 perplexity는 $2^{\sum_j -p_{j|i}log_2(p_{j|i})}$가 됩니다. 결과적으로 확률분포의 각 값이 비슷해질수록 높은 entropy를 가지게 되고, entropy가 높아질수록 높은 perplexity를 가지게 됩니다.

이제 원래 SNE에서의 perplexity로 돌아갑시다. SNE에서 perplexity는 정해주는 하이퍼 파라미터입니다. 만약 SNE를 할 때, 데이터 객체 $x_i$로부터 거리가 멀더라도 비슷한 확률로 유사도가 표현되도록 하고 싶다면 높은 perplexity를 정해주면 될 것입니다. 쉽게 말해 데이터 객체 $x_i$의 이웃의 범위가 넓어지는 셈이죠. 만약 perplexity가 낮다면 이웃의 범위가 좁아지는 것이 될테구요(*하지만 SNE의 원 논문을 보면, SNE의 결과는 5~50의 perplexity의 변화에는 robust하게 일정하다고 합니다)*.

어쨌든, 높은 perplexity를 정해주면 높은 entropy가 결정되고 이는 곧 $P_i$가 고른 것, 다시 말해 $x_i$로부터 거리가 먼 객체와 작은 객체간의 확률 차이가 작음을 의미함을 이해하셨을 것입니다. 이제 $p_{j|i}$에 들어간 $\sigma_i$에 대해서 생각해봅시다. 결국 perplexity가 정해지면 entropy가 정해지고, entropy가 정해지면 $p_{j|i}$가 정해져야 되기 때문에 결과적으로 $x_i$마다 $\sigma_i$가 정해집니다. 그리고 $\sigma_i$의 역할에 대해 생각해보면, $\sigma_i$가 커질수록, 분자인 유클리디언 거리, 다시 말해 객체 간의 거리가 무의미해짐을 알 수 있습니다. 다시 말해 확률분포 $P_i$가 고르게 되는 것이죠. 결론적으로 perplexity를 높게 잡으면 $\sigma_i$가 커지고, 낮게 잡으면 $\sigma_i$가 작아집니다. 결국 perplexity를 정의해주는 것은 데이터 객체마다 그 주변의 밀도가 다른 점을 반영하려는 의도로 볼 수 있습니다.

이제 다시 cost function으로 돌아옵시다. 드디어, cost function 계산을 위한 준비가 끝났습니다. $x_i$는 데이터셋에서 주어지는 값임을 알고, $y_i$는 우리가 최적화 문제를 통해 찾아야 하는 값인 것도 알고 있죠. 그리고 이제 $\sigma_i$를 계산하기 위해 각 데이터 객체 $x_i$ 별로 perplexity를 정해줘야 할 것입니다. 그럼 이제 cost function을 minmize하는 $y_i$를 어떻게 찾아야 할까요? 논문의 저자들은 gradient descent 방법을 활용합니다. 때문에 $y_i$를 처음에 random하게(isotropic Gaussian 분포에서) generate 하고, 이를 gradient descent에 momentum term을 추가하여 update합니다. t-SNE 논문에서 제시한 gradient와 solution의 update 식은 아래와 같습니다. 아래 식에서 $Y_{(t)}$는 t시점의 solution 행렬이고, $\frac{\partial C}{\partial Y^{(t)}}$가 gradient이며, $\eta$는 학습률, $\alpha^{(t)}$는 momentum term입니다. 
<p align="center"><img src="https://user-images.githubusercontent.com/112034941/195265268-754e005b-f781-4151-ab6c-0c99e1777848.png" height="180px" width="600px"></p>

그런데 조금 이상하지 않으신가요? 실제로 모멘텀 경사하강법의 solution update 식을 보면 $- \eta *$(gradient)로 되어있는데, 지금 논문에서의 식은 덧셈으로 연결되어있습니다. 제 생각에는, 실제로 이 cost function의 gradient를 구하는 과정을 보면 어차피 gradient를 빼줘야 함을 고려하여 gradient 식의 $(y_i-y_j)$ 부분이 원래는 $(y_j-y_i)$였는데 여기에 -를 곱한 것으로 보입니다. gradient를 구하는 자세한 과정은 여기서는 생략하겠습니다. 앞서 언급했던 강필성 교수님 [유튜브 강의](https://www.youtube.com/watch?v=ytRmxBvyGG0&list=PLetSlH8YjIfWMdw9AuLR5ybkVvGcoG2EW)에 과정이 나와있으니 참고하시면 좋을 것 같습니다. 

여기까지 t-SNE를 위한 핵심아이디어를 제공한 SNE를 알아보았습니다. 이제, t-SNE에서는 이 아이디어를 어떻게 활용했는지 살펴보겠습니다.

### t-distribution Stochastic Neighbor Embedding(t-SNE)
t-SNE의 저자들은 SNE에 두 가지 아이디어를 추가합니다. 첫번째는 symmetric SNE이고 두번째는 t-distribution의 도입입니다. 각각 $P_i$와 $Q_i$에 영향을 미칩니다. 우선 symmetric SNE부터 살펴보겠습니다.
#### symmetric SNE
본래 SNE에서는 $p_{i|j}$와 $p_{j|i}$가 서로 다를 수 있습니다. 각 데이터 객체별로 $\sigma_i$가 정해지기 때문이죠. t-SNE의 저자들은 데이터 객체 i, j에 대해 동일한, 다시말해 대칭적인 확률분포를 생각하였습니다. 논문에서 제시된 것은 두가지로 아래와 같습니다. 
<p align="center"><img src="https://user-images.githubusercontent.com/112034941/195279421-87d362f4-235d-4b70-bf85-ea3a66b31ee5.png" height="200px" width="600px"></p>

논문에서 선택한 것은 오른쪽입니다. 왼쪽의 경우 한번만 계산해도 되기 때문에 계산도 효율적이고, 더 분명한 의미를 가지지만 한가지 문제가 존재합니다. 바로 이상치가 발생할 경우, 해당 이상치는 다른 점들에 비해 모두 멀리 떨어져 있으므로 $p_{ij}$가 작게 되고, 결국 이 이상치의 mapping이 불분명해지게 됩니다. 그래서 오른쪽과 같이 정의하게 되면 어떤 경우에도 $\sum_j p_{ij} > \frac{1}{2n}$ 으로 만들어주기 때문에 앞서 언급했던 이상치의 문제를 어느정도 해결할 수 있습니다. 논문의 저자들은 이 symmetric SNE가 원래 SNE에 비해 동일하거나 그 이상의 성능을 보임을 확인했다고 합니다. 

#### Employment of t-distribution
symmetric SNE에서의 아이디어를 통해 $p_{ij}$를 정의한 후, 저자들은 crowding problem을 해결하기 위한 목적으로 $q_{ij}$를 아래와 같이 정의합니다. 
<p align="center"><img src="https://user-images.githubusercontent.com/112034941/195288693-1c158327-4c13-4f7a-99db-32c60934dbb9.png" height="100px" width="300px"></p>

symmetric SNE에서 $q_{ij}$의 분포는 정규분포를 사용하여 얻어집니다. 그런데, 정규분포에서는 평균값에서 멀어질수록, 즉 여기서는 객체 i로부터의 거리가 멀어질수록 그 확률함수의 값이 급격하게 낮아집니다. 이러한 경우 실제로는 적당한(moderate) 거리에 있는 객체들이 들어갈 영역이 작은 문제가 생기는데, 이를 crowding problem이라 합니다. 때문에 저자들은 $q_{ij}$를 정규분포가 아닌 자유도가 1인 t-분포(코시 분포와 동일합니다)를 사용하여 위 식과 같이 정의합니다. t-분포는 정규분포와 다르게(자유도가 1이므로) 앞서 말한 것에 비하여 적당한 거리의 객체들이 저차원 공간에 mapping될 영역을 만들어주어 crowding problem을 해결하는 것에 일조합니다.

#### t-SNE
결론적으로 위 두가지 아이디어, symmetric 한 $p_{ij}$의 정의와 t-분포를 이용한 $q_{ij}$의 정의를 기존 SNE에 도입한 SNE를 t-SNE라고 합니다. 따라서 t-SNE에서의 원래 공간에서의 두 객체간의 유사도 $p_{ij}$, 차원 축소 결과 두 객체간의 유사도 $q_{ij}$, 이 때의 cost function, 그리고 gradient를 정리하면 아래와 같습니다. 
<p align="center"><img src="https://user-images.githubusercontent.com/112034941/195350179-54189798-b73d-4dc8-a013-4f53c49fc7ec.png" height="300px" width="1000px"></p>

그리고 이에 따라 t-SNE 논문에서 제시된 t-SNE 수도코드는 아래와 같습니다.
<p align="center"><img src="https://user-images.githubusercontent.com/112034941/195351628-82f1b0c2-7628-43a5-aa8b-9814efbaa4c4.png"></p>

정리하면, t-SNE 알고리즘의 hyper-parameter는 perplexity, iteration의 수, learning rate, momentum이며, 알고리즘은 hyper-parameter 설정 이후 $p_{j|i}$ 계산, 초기해 initialize (generated from gaussian), gradient descent의 순서로 진행됩니다. 

이제, 위 알고리즘을 실제로 구현해 보겠습니다.

---

## t-SNE Implementation
t-SNE 알고리즘의 순서부터 생각해봅시다. 하이퍼 파라미터의 설정 이후에는 1) $p_{j|i}$를 계산(전체 객체 n개에 대해), 2) $p_{ij}$ 계산, 3) 초기해 설정, 4) gradient 계산, 5) solution update, 6) 이후 t번 4, 5) 반복의 순서로 이뤄져야 합니다. 하지만 실제로 이를 구현하기 위해서는 1)을 조금 더 깊게 파고 들어야합니다. 앞서 살펴본 t-SNE의 개념을 되짚어 보면, $p_{j|i}$를 계산하기 위해서는 각 객체 사이의 유클리드 거리 계산 및 perplexity에 따른 각 데이터 객체 별 $\sigma_i$를 도출하는 과정이 선행되어야 함을 알 수 있습니다. $\sigma_i$ 도출을 위해 흔히 사용하는 알고리즘은 binary search 입니다. 이진탐색의 개념을 자세히 짚고 넘어가지는 못하지만, 최대한 압축하여 설명하자면 여기서의 이진탐색은 0부터 최대 $\sigma_i$ 중간값의 $\sigma_i$를 구해 대입해본 후 원하는 perplexity 보다 낮으면 0과 현재 $\sigma_i$ 사이의 값을 넣어보고 높으면 현재 $\sigma_i$와 최대 $\sigma_i$ 사이의 값을 넣어보는 것을 반복하며 perplexity를 만족하는 $\sigma_i$를 찾는 식으로 찾아내는 것을 말합니다. 

위 과정을 생각했을 때, 구현해주어야 할 함수는 다음과 같이 정할 수 있습니다.
1. euclidean distance matrix 반환

하나하나 살펴봅시다. 



