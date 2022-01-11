---
layout: single
title: "A comprehensive introduction to different types of convolutions in deep learning[번역]"
categories: [UHRGAN]
tag: [UHRGAN, CNN]
---

url: 

[](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215)

## 서론

이 글은 2D / 3D / 1x1 / Transposed / Dilated / Spatially Separable / Depthwise Separable / Flattened / Grouped / Shuffled Grouped Convolution 등 딥러닝의 여러 종류의 convolutions 를 헷갈려하는 당신에게 도움이 될 것이다.

이 글에서는 딥러닝에 많이 사용되는 몇가지 종류의 convolution을 모든 사람들이 이해하기 쉽게끔 정리를 해놓았다. 이 글 이외에 정리가 잘된 다른 글들이 아래 리스트에 있으니 읽어보는것을 추천한다. 

이 글이 읽는 사람들이 convolution에 대한 직관력 향상에 도움이 되어 하고있는 공부 또는 연구에 유용하게 참고가 되었으면 좋겠다. 댓글과 제안은 언제든 환영한다. 

## 목차:

1. Convolution vs Cross-correlation
2. 딥러닝에서의 Convolution (Single channel version, multi-channel version)
3. 3D Convolution
4. 1 x 1 Convolution
5. Convolution Arithmetic
6. Transposed Convolution (Deconvolution, checkerboard artifacts)
7. Dilated Convolution (Atrous Convolution)
8. Separable Convolution (Spatially Separable Convolution, Depthwise Convolution)
9. Flattened Convolution
10. Grouped Convolution
11. Shuffled Grouped Convolution
12. Pointwise Grouped Convolution

## 1. Convolution vs Cross-correlation

Convolution은 신호처리, 이미지 처리 그리고 다른 엔지니어링 / 과학 분야에 많이 사용이 되는 기법이다. 딥러닝에서는 이 기법을 사용한 모델은 Convolutional Neural Network (CNN)이라고 불린다. 하지만, 딥러닝에서 convolution은 신호/이미지 처리에서 기본적으로 cross-correlation과 같다.

너무 자세한 내용을 다루지 않는 선에서 설명을 한다면 신호/이미지 처리에서 convolution은 다음과 같은 식으로 정의가 될 수 있다. 

![Untitled.png](../../images/2021-01-11-A_Comprehensive_Introduction_to_Different_Types_of_Convolutions_in_Deep_Learning_Translated/Untitled.png)

위 수식을 보면 두 함수 곱의 적분에서 하나의 함수는 역전(y축)되어 x축 위에서 이동이 되는 것이 정의가 된다. 이것은 아래 그림을 통해 더 자세히 설명을 할 수 있다.

![신호처리에서의 Convolution. 필터 g는 역전이 되어있고 x축 위에서 이동하고 있다. 매 위치마다 함수 f와 g로 감싸진 면적을 구한다. 이 면적은 해당 위치의 convolution의 값이다. 이미지의 출처는 해당 [링크](http://fourier.eng.hmc.edu/e161/lectures/convolution/index.html)이며 수정 후 사용.](../../images/2021-01-11-A_Comprehensive_Introduction_to_Different_Types_of_Convolutions_in_Deep_Learning_Translated/Untitled%201.png)

신호처리에서의 Convolution. 필터 g는 역전이 되어있고 x축 위에서 이동하고 있다. 매 위치마다 함수 f와 g로 감싸진 면적을 구한다. 이 면적은 해당 위치의 convolution의 값이다. 이미지의 출처는 해당 [링크](http://fourier.eng.hmc.edu/e161/lectures/convolution/index.html)이며 수정 후 사용.

여기서 함수 g는 필터 역할이다. 함수 g는 역전이 되어있고 x축 위에서 이동하고 있다. 매 위치마다 함수 f와 역전된 함수 g로 감싸진 면적을 구한다. 이 면적은 해당 위치의 convolution의 값이다. 

반면에 corss-correlation은 sliding dot product 또는 두 함수의 sliding inner-product으로 알려져있다. cross-correlation에서 필터 함수는 역전이 되어있지 않다. 대신 그대로 함수 f를 이동하며 지나간다. 마찬가지로 여기서 두 함수 f와 g로 감싸진 면적이 cross-correlation의 값이 된다. 아래의 그림은 correlation과 cross-correlation의 차이를 나타내고 있다.

![신호처리에서 Convolution과 Cross-correlation의 차이. 이미지의 출처는 해당 [링크](https://en.wikipedia.org/wiki/Convolution)이며 수정 후 사용.](../../images/2021-01-11-A_Comprehensive_Introduction_to_Different_Types_of_Convolutions_in_Deep_Learning_Translated/Untitled%202.png)

신호처리에서 Convolution과 Cross-correlation의 차이. 이미지의 출처는 해당 [링크](https://en.wikipedia.org/wiki/Convolution)이며 수정 후 사용.

딥러닝에서는 convolution에서 필터는 역전이 되지 않는다. 엄밀히 말하면, convolution이 아니라 cross-correlation이라고 해야한다. 왜냐하면 element-wise multiplication과 addition이 수행이 되기 때문이다. 하지만, 딥러닝에서는 그냥 convolution이라고 부른다. 이러한 부분은 실제로 문제가 되지 않는다. 왜냐하면 필터의 가중치는 training 중 학습이 되기 때문이다. 만약에 예제의 역전된 g가 올바른 함수라면 학습 후 학습된 필터는 역전된 함수 g와 같게 될 것이다. 따라서 실제 convolution처럼 train 전에 필터를 역전할 필요가 없다.

## 2. 딥러닝에서의 Convolution

Convolution을 하는 목적은 input에서 유용한 feature를 추출하기 위함이다. 이미지 처리에서 사용자가 선택할 수 있는 필터의 종류는 많다. 각 종류의 필터는 input의 이미지에서 서로 다른 관점의 feature를 추출할 수 있게 해준다 (horizontal / vertical / diagonal edges). Convolutional Neural Network에서는 이와 비슷하게 input 이미지에서 다양한 features들이 여러개의 필터들을 통해 추출이 될 수 있는데 이 필터들은 train 중 가중치가 자동으로 학습이 된다. 그리고 추출된 features 들을 종합적으로 이용하여 결정을 내린다. 

Convolution을 사용을 하면 여러가지 장점이 있다. 예를들어 weights sharing(가중치 공유)과 translation invariant이 있다. 또한, convolution은 픽셀의 공간 관계를 고려한다. 이러한 부분에 있어서 convolution은  computer vision task에 많은 도움이 된다. 왜냐하면 해당 task에서는 특정 component 가 다른 component와의 공간적 관계를 다루기 때문이다. 

## 2.1. Convolution: the single channel version

딥러닝에서 convolution은 element-wise multiplication과 addition을 수행한다. 1개의 channel을 가지고 있는 이미지에서 convolution은 다음과 같이 설명이 될 수 있다. 

![단일 channel에서의 Convolution. 이미지의 출처는 해당 [링크](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)이다. ](../../images/2021-01-11-A_Comprehensive_Introduction_to_Different_Types_of_Convolutions_in_Deep_Learning_Translated/test.gif)

단일 channel에서의 Convolution. 이미지의 출처는 해당 [링크](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)이다. 

![단일 channel에서의 Convolution. 이미지의 출처는 해당 [링크](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)이다. ](../../images/2021-01-11-A_Comprehensive_Introduction_to_Different_Types_of_Convolutions_in_Deep_Learning_Translated/conv.gif)

단일 channel에서의 Convolution. 이미지의 출처는 해당 [링크](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)이다. 

여기에서 사용된 필터는 3 x 3 행렬이고 원소는 [[0,1,2], [2,2,0], [0,1,2]] 의 값을 가지고 있다. 필터는 input 위를 지나가며 각 위치에서 element-wise multiplication과 addition을 수행을 하고 있다. 각 위치에서 성분곱(아다마르 곱)과 합에서 하나의 값이 나온다. 따라서 최종 결과는 3 x 3 이 나오게 된다(위 예제에서는 stride = 1, padding = 0 이고 이에대한 설명은 아래 설명 예정).