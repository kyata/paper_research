# High-Resolution Image Synthesis with Latent Diffusion Models



## Authors

- Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser Bjo ̈rn Ommer

- Ludwig Maximilian University of Munich & IWR Heidelberg University
- Germany Runway ML

## Abstract

```
By decomposing the image formation process into a se- quential application of denoising autoencoders, diffusion models (DMs) achieve state-of-the-art synthesis results on image data and beyond. Additionally, their formulation allows for a guiding mechanism to control the image generation process without retraining.
However, since these models typically operate directly in pixel space, optimiza- tion of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evalu- ations. To enable DM training on limited computational resources while retaining their quality and flexibility, we apply them in the latent space of powerful pretrained au- toencoders. In contrast to previous work, training diffusion models on such a representation allows for the first time to reach a near-optimal point between complexity reduc- tion and detail preservation, greatly boosting visual fidelity. By introducing cross-attention layers into the model archi- tecture, we turn diffusion models into powerful and flexible generators for general conditioning inputs such as text or bounding boxes and high-resolution synthesis becomes pos- sible in a convolutional manner. Our latent diffusion models (LDMs) achieve a new state of the art for image inpaint- ing and highly competitive performance on various tasks, including unconditional image generation, semantic scene synthesis, and super-resolution, while significantly reducing computational requirements compared to pixel-based DMs.
```

拡散モデル（DM）は、画像形成プロセスをノイズ除去オートエンコーダの逐次適用に分解することで、画像データおよびそれ以外のデータに対して最先端の合成結果を実現する。さらに、その定式化により、再学習なしで画像生成プロセスを制御するためのガイドメカニズムが可能になる。しかし、これらのモデルは通常ピクセル空間で直接動作するため、強力なDMの最適化にはGPUで数百日を費やすことが多く、逐次評価による推論も高価である。DMの品質と柔軟性を維持したまま、限られた計算資源でDMのトレーニングを可能にするために、我々は強力な事前訓練されたautoencoderの潜在空間にDMを適用する。従来の研究とは異なり、このような表現で拡散モデルを学習することで、初めて複雑さの軽減と細部の保存の間の最適に近い点に到達し、視覚的忠実度を大幅に向上させることができる。また、クロスアテンションレイヤーをモデル構造に導入することで、拡散モデルをテキストやバウンディングボックスのような一般的な条件入力に対する強力で柔軟なジェネレーターに変え、畳み込み方式で高解像度合成を可能にしました。我々の潜在拡散モデル（LDM）は、ピクセルベースのDMと比較して計算量を大幅に削減しながら、無条件の画像生成、意味的なシーン合成、超解像などの様々なタスクにおいて、画像インペイントの新しい技術水準と高い競争力を達成しました。

- 拡散モデル(Diffusion Models)はノイズ除去オートエンコーダの逐次適用に分解して画像を生成する
- 定式化すると再学習なしで画像生成プロセスを制御することが可能？
- 通常ピクセル空間で直接動作するため、推論および、モデルの最適化に計算リソースを要する(GPU/ XXhundrets days)　既知問題
- 品質と柔軟性を維持しながら、限られた計算資源でトレーニングできるようにするために、潜在空間にDMを適用可能にした
- これにより、複雑さを軽減し、細部の保存の間(画像の?)において、視覚的忠実度を大幅に向上させることができる。
- Cross-Attnレイヤをモデルに導入
- TXTやBBXのような一般的な入力条件で、DMを
- 畳み込み方式での高解像度合成を可能にした。
- 潜在拡散モデルは、ピクセルベースのDMと比較して、以下のメリットがある
  - 計算量の大幅な削減
  - 条件なしでの画像生成
  - セマンティックシーン合成

## 1. Introduction

```
Image synthesis is one of the computer vision fields with the most spectacular recent development, but also among those with the greatest computational demands. Espe- cially high-resolution synthesis of complex, natural scenes is presently dominated by scaling up likelihood-based mod- els, potentially containing billions of parameters in autore- gressive (AR) transformers [61,62]. In contrast, the promis- ing results of GANs [3, 24, 36] have been revealed to be mostly confined to data with comparably limited variability as their adversarial learning procedure does not easily scale to modeling complex, multi-modal distributions. Recently, diffusion models [77], which are built from a hierarchy of denoising autoencoders, have shown to achieve impressive results in image synthesis [27,80] and beyond [6,41,44,52], and define the state-of-the-art in class-conditional image synthesis [14,28] and super-resolution [67]. Moreover, even unconditional DMs can readily be applied to tasks such as inpainting and colorization [80] or stroke-based syn- thesis [48], in contrast to other types of generative mod- els [17, 42, 64]. Being likelihood-based models, they do not exhibit mode-collapse and training instabilities as GANs and, by heavily exploiting parameter sharing, they can model highly complex distributions of natural images with- out involving billions of parameters as in AR models [62].
```

画像合成は、コンピュータビジョン分野の中で最も目覚しい発展を遂げている分野の一つであるが、同時に**最も計算量の多い分野の一つでもある。**

特に、複雑で自然なシーンの高解像度合成は、現在、尤度ベースのモデルのスケールアップが主流であり、自己表現（AR）変換器に数十億のパラメータを含む可能性がある[61, 62]。

一方、GAN[3, 24, 36]の有望な結果は、その敵対的学習手順が複雑なマルチモーダル分布のモデリングに容易に拡張できないため、ほとんどが比較的に限定された変動性を持つデータに限られていることが明らかにされている。

最近では、ノイズ除去オートエンコーダの階層から構築される拡散モデル [77] が、画像合成 [27,80] 以降 [6,41,44,52] で印象的な結果を達成し、クラス条件付き画像合成 [14,28] や超解像 [67] の最先端を定義していることが示されている。

さらに、他のタイプの生成モデル[17, 42, 64]とは対照的に、無条件のDMでさえ、インペインティングやカラー化[80]、ストロークベースの合成[48]といったタスクに容易に適用することができる。尤度ベースのモデルであるため、GANのようなモード崩壊や学習の不安定性がなく、パラメータの共有を多用することで、ARモデルのように何十億ものパラメータを必要とせず、自然画像の非常に複雑な分布をモデル化できる [62] 。

```
Democratizing High-Resolution Image Synthesis DMs belong to the class of likelihood-based models, whose mode-covering behavior makes them prone to spend excessive amounts of capacity (and thus compute resources) on modeling imperceptible details of the data [15, 68]. 
Although the reweighted variational objective [27] aims to address this by undersampling the initial denoising steps, DMs are still computationally demanding, since training and evaluating such a model requires repeated function evaluations (and gradient computations) in the high-dimensional space of RGB images.

As an example, training the most powerful DMs often takes hundreds of GPU days (e.g. 150 - 1000 V100 days in [14]) and repeated evaluations on a noisy version of the input space render also inference expensive, so that producing 50k samples takes approximately 5 days [14] on a single A100 GPU.

This has two consequences for the research community and users in general: Firstly, train- ing such a model requires massive computational resources only available to a small fraction of the field, and leaves a huge carbon footprint [59, 81].

Secondly, evaluating an already trained model is also expensive in time and memory, since the same model architecture must run sequentially for a large number of steps (e.g. 25 - 1000 steps in [14]).
```

- 高解像度画像合成DMを民主化することは尤度ベースモデルのクラスに属し、そのモードカバー動作はデータの知覚できない細部をモデル化するために過剰な容量（したがって計算資源）を費やす傾向がある[15, 68]。

- 再重み付け変分最適化？ [27] は、最初のノイズ除去ステップをアンダーサンプリングすることでこの問題に対処することを目的としているが、このようなモデルの学習と評価には、RGB画像の高次元空間における関数評価（および勾配計算）の繰り返しが必要となるため、DMは依然として計算負荷が高い。

- 例えば、最も強力なDMのトレーニングには数百GPU日（例えば[14]では150～1000V100日）かかることが多く、入力空間のノイズの多いバージョンで繰り返し評価することで推論も高くなり、50kサンプルを作成するのにA100 GPU1台で約5日かかる[14]。
- このことは、研究コミュニティと一般ユーザーにとって2つの結果をもたらす： 
  - 第一に、このようなモデルの学習には、研究分野のごく一部にしか利用できない膨大な計算資源が必要であり、膨大なカーボンフットプリントが残ります[59, 81]。
  - 第二に、学習済みのモデルを評価する場合、同じモデル・アーキテクチャを多数のステップ（例えば[14]では25～1000ステップ）で順次実行しなければならないため、時間とメモリが高価になることである。

### Departure to Latent Space

```
Our approach starts with the analysis of already trained diffusion models in pixel space: Fig. 2 shows the rate-distortion trade-off of a trained model. As with any likelihood-based model, learning can be roughly divided into two stages: First is a perceptual compression stage which removes high-frequency details but still learns little semantic variation. In the second stage, the actual generative model learns the semantic and conceptual composition of the data (semantic compression). We thus aim to first find a perceptually equivalent, but compu- tationally more suitable space, in which we will train diffu- sion models for high-resolution image synthesis.
```

私たちのアプローチは、画素空間ですでに訓練された拡散モデルの分析から始まります： 図2は、学習済みモデルのレートとディストーションのトレードオフを示したものです。

他の尤度ベースモデルと同様に、学習は大きく2つの段階に分けられます

- 第一段階は知覚的圧縮で、高周波の細部を取り除くが、意味的な変化はほとんど学習されない。
- 第二段階では、実際の生成モデルがデータの意味的・概念的な構成を学習する（意味的圧縮）。
- このように、我々はまず、高解像度画像合成のためのディフュージョンモデルを訓練するために、**知覚的には同等だが、計算上はより適切な空間を見つけることを目的としている。**

```
Following common practice [10, 21, 61, 62, 90], we sep- arate training into two distinct phases: First, we train an autoencoder which provides a lower-dimensional (and thereby efficient) representational space which is perceptu- ally equivalent to the data space. Importantly, and in con- trast to previous work [21,61], we do not need to rely on ex- cessive spatial compression, as we train DMs in the learned latent space, which exhibits better scaling properties with respect to the spatial dimensionality. The reduced complex- ity also provides efficient image generation from the latent space with a single network pass. We dub the resulting model class Latent Diffusion Models (LDMs).
```

一般的な方法 [10, 21, 61, 62, 90] に従って、我々はトレーニングを2つの異なるフェーズに分離する： まず、データ空間と知覚的に等価な低次元の表現空間を提供するオートエンコーダを訓練する。

- 重要なことは、先行研究[21,61]とは対照的に、空間次元に対してより優れたスケーリング特性を示す学習済み潜在空間でDMを訓練するため、過剰な空間圧縮に頼る必要がないことである。
- また、複雑さが軽減されているため、1回のネットワークパスで潜在空間から効率的に画像を生成することができる。
  この結果、潜在拡散モデル(LDM)と呼ばれるモデルクラスが誕生した。

```
A notable advantage of this approach is that we need to train the universal autoencoding stage only once and can therefore reuse it for multiple DM trainings or to explore possibly completely different tasks [76]. This enables effi- cient exploration of a large number of diffusion models for various image-to-image and text-to-image tasks. For the lat- ter, we design an architecture that connects transformers to the DM’s UNet backbone [66] and enables arbitrary types of token-based conditioning mechanisms, see Sec. 3.3.
```

- このアプローチの特筆すべき利点は、普遍的な自動符号化ステージを一度だけ訓練する必要があるため、複数のDM訓練や、全く異なるタスクの探索に再利用できることである[76]。
- これにより、様々な画像から画像、テキストから画像へのタスクに対して、多数の拡散モデルの効率的な探索が可能になる。
- 後者については、トランスフォーマーをDMのUNetバックボーン[66]に接続し、任意のタイプのトークンベースの条件付けメカニズムを可能にするアーキテクチャを設計する（セクション3.3参照）。

### Contributions

```
Besides providing a step in the direction of “democratizing” research on DMs, our work contains the following contributions:

(i) In contrast to purely transformer-based approaches [21, 61], our method scales more graceful to higher dimensional data and can thus (a) work on a compression level which provides more faithful and detailed reconstructions than previous work (see Fig. 1) and (b) can be efficiently applied to high-resolution synthesis of megapixel images.

(ii) We achieve competitive performance on multiple tasks (unconditional image synthesis, inpainting, stochastic super-resolution) and datasets while significantly lowering computational costs. Compared to pixel-based diffusion approaches, we also significantly decrease inference costs.

(iii) We show that, in contrast to previous work [87] which learns both an encoder/decoder architecture and a score-based prior simultaneously, our approach does not require a delicate weighting of reconstruction and generative abilities.
This ensures extremely faithful reconstructions and requires very little regularization of the latent space.

(iv) We find that for densely conditioned tasks such as super-resolution, inpainting and semantic synthesis, our model can be applied in a convolutional fashion and render large, consistent images of ∼ 10242 px.

(v) Moreover, we design a general-purpose conditioning mechanism based on cross-attention, enabling multi-modal training. We use it to train class-conditional, text-to-image and layout-to-image models.

(vi) Finally, we release pretrained latent diffusion and autoencoding models at https://github.com/CompVis/latent-diffusion which might be reusable for a various tasks besides training of DMs [76].
```

DMの研究を「民主化」する方向への一歩を提供することに加え、我々の研究は以下のような貢献をする：

(i) 純粋な変換器ベースのアプローチ[21, 61]とは対照的に、本手法は高次元のデータに対してより優雅にスケールするため、**(a) 従来の研究よりも忠実で詳細な再構成を提供する圧縮レベルで動作し（図1参照） (b) メガピクセル画像の高解像度合成に効率的に適用することができる。**

(ii) **計算コストを大幅に低減しながら、複数のタスク（無条件画像合成、インペインティング、確率的超解像）およびデータセットにおいて競争力のある性能を達成する。**また、**ピクセルベースの拡散アプローチと比較して、推論コストを大幅に削減する。**

(iii) エンコーダ/デコーダアーキテクチャとスコアベースの事前学習を同時に行う先行研究 [87] とは対照的に、本アプローチでは再構成能力と生成能力の微妙な重み付けを必要としないことを示す。これにより、極めて忠実な再構成が保証され、潜在空間の正則化もほとんど必要ない。

(iv) 超解像、インペインティング、意味合成のような高密度な条件を持つタスクに対して、本モデルを畳み込み方式で適用し、1024^2px程度の大規模で一貫した画像をレンダリングできることを見出すことができた。

(v) さらに、**クロスアテンションに基づく汎用的な条件付け機構を設計し、マルチモーダルな学習を可能にした**。これを用いて、クラス条件付けモデル、テキストから画像への変換モデル、レイアウトから画像への変換モデルを学習する。

(vi) 最後に、訓練済みの潜在拡散モデルと自動符号化モデルを https://github.com/CompVis/latent-diffusion で公開する。これは、DMの訓練以外にも様々なタスクに再利用できる可能性がある[76]。

## 2. Related Work

T.B.W

## 3. Method

```
To lower the computational demands of training diffusion models towards high-resolution image synthesis, we observe that although diffusion models allow to ignore perceptually irrelevant details by undersampling the corresponding loss terms [27], they still require costly function evaluations in pixel space, which causes huge demands in computation time and energy resources.

We propose to circumvent this drawback by introducing an explicit separation of the compressive from the generative learning phase (see Fig. 2).
To achieve this, we utilize an autoencoding model which learns a space that is perceptually equivalent to the image space, but offers significantly reduced computational complexity.

Such an approach offers several advantages: 

(i) By leaving the high-dimensional image space, we obtain DMs which are computationally much more efficient because sampling is performed on a low-dimensional space. 

(ii) We exploit the inductive bias of DMs inherited from their UNet architecture [66], which makes them particularly effective for data with spatial structure and therefore alleviates the need for aggressive, quality-reducing compression levels as required by previous approaches [21, 61].

(iii) Finally, we obtain general-purpose compression models whose latent space can be used to train multiple generative models and which can also be utilized for other downstream applica- tions such as single-image CLIP-guided synthesis [23].

```

高解像度画像合成のために拡散モデルを学習する際の計算量を減らすために、拡散モデルは対応する損失項をアンダーサンプリングすることで知覚的に無関係な細部を無視することができますが[27]、ピクセル空間での高価な関数評価を必要とし、計算時間やエネルギー資源に大きな需要があることが分かっています。

我々は、この欠点を回避するために、**圧縮学習と生成学習を明示的に分離することを提案する（図2参照）。**
これを実現するために、**画像空間と知覚的に等価な空間を学習する自動エンコードモデルを利用し、計算量を大幅に削減する**ことができる。

このようなアプローチには以下、複数の利点がある

1. 高次元の画像空間から離れることで、低次元空間でのサンプリングとなるため、計算効率が格段に向上したDMが得られる。
2. UNetアーキテクチャ[66]から継承されたDMの帰納的バイアスを利用し、空間構造を持つデータに対して特に有効であるため、従来のアプローチ[21, 61]で必要とされた積極的で品質を低下させる圧縮レベルの必要性を緩和している。
3. 最後に、潜在空間が複数の生成モデルの学習に使用でき、**単一画像CLIPガイド付き合成[23]などの他の下流アプリケーションにも利用できる汎用圧縮モデルを得ることができた。**

### 3.1 Perceptual Image Compression

```
Our perceptual compression model is based on previous work [21] and consists of an autoencoder trained by combination of a perceptual loss [99] and a patch-based [29] adversarial objective [18, 21, 96].
This ensures that the reconstructions are confined to the true image manifold by enforcing local realism and avoids bluriness introduced by relying solely on pixel-space losses such as L2 or L1 objectives.
```

- 我々の知覚圧縮モデルは、以前の研究 [21] に基づいており、perceptual loss [99] とパッチベース [29] のadversarial objective [18, 21, 96] を組み合わせて訓練したオートエンコーダで構成されている。
- これにより、局所的なリアリズムを強制することで再構成が真の画像多様体に限定されることを保証し、L2やL1目的などのピクセル空間損失のみに依存することで生じるぼやけを回避している。

```
More precisely, given an image x ∈ RH×W×3 in RGB space, the encoder E encodes x into a latent representation z = E(x), and the decoder D reconstructs the image from the latent, giving x ̃ = D(z) = D(E(x)), where z ∈ Rh×w×c.
Importantly, the encoder downsamples the image by a factor f = H/h = W/w, and we investigate different downsampling factors f = 2m, with m ∈ N.
```

より正確には、

- RGB空間の画像$x \in \mathbb{R}^{H×W×3}$が与えられたとき、エンコーダ$ \epsilon $は$x$を潜在表現 $z=\epsilon(x)$に符号化する

- デコーダDは潜在表現から画像を再構成し、$\tilde{x} = D(z)＝D(E(x))$とする
  - ここで, $z \in \mathbb{R}^{h \times w \times h}$である
- 重要なのは、エンコーダーが画像を係数$f = H/h = W/w$によってダウンサンプルすることだ
  - 我々はダウンサンプル係数$f = 2^m with m \in \mathbb{N}$で違いを調査した

```
In order to avoid arbitrarily high-variance latent spaces, we experiment with two different kinds of regularizations.
The first variant, KL-reg., imposes a slight KL-penalty towards a standard normal on the learned latent, similar to a VAE [42, 64], whereas VQ-reg, uses a vector quantization layer [90] within the decoder.
This model can be interpreted as a VQGAN [21] but with the quantization layer absorbed by the decoder. Because our subsequent DM is designed to work with the two-dimensional structure of our learned latent space z = E(x), we can use relatively mild compres- sion rates and achieve very good reconstructions.

This is in contrast to previous works [21, 61], which relied on an arbitrary 1D ordering of the learned space z to model its distribution autoregressively and thereby ignored much of the inherent structure of z.
Hence, our compression model preserves details of x better (see Tab. 1). The full objective and training details can be found in the supplement.
```

- 任意に**高変量な潜在空間を回避するために、2種類の正則化を実験した。**

  - KL-reg

  - VQ-reg

- 最初のバリエーションであるKL-reg.は、VAE [42, 64]と同様に、学習した潜在能力に対して標準正規に対するわずかなKL-penaltyを課すが、VQ-regはデコーダ内でベクトル量子化層 [90] を使用する。

  - このモデルはVQGAN[21]と解釈できるが、量子化層がデコーダーに吸収されている。
    この後のDMは学習した潜在空間z = E(x)の2次元構造を扱うように設計されているため、比較的穏やかな圧縮率を使用し、非常に優れた再構成を達成することが可能である。
  - これは、学習空間zの任意の1次元順序に依存してその分布を自己回帰的にモデル化し、それによって$z$の固有の構造の多くを無視していた先行研究[21, 61]とは対照的である。
  - したがって、我々の圧縮モデルは$x$の詳細をよりよく保存している（表1参照）。
  - 目的およびトレーニングの詳細については、付録を参照。

### 3.2. Latent Diffusion Models

```
Diffusion Models [77] are probabilistic models designed to learn a data distribution p(x) by gradually denoising a nor- mally distributed variable, which corresponds to learning the reverse process of a fixed Markov Chain of length T. For image synthesis, the most successful models [14,27,67] rely on a reweighted variant of the variational lower bound on p(x), which mirrors denoising score-matching [80]. 
These models can be interpreted as an equally weighted sequence of denoising autoencoders εθ (xt , t); t = 1 . . . T , which are trained to predict a denoised variant of their input xt, where xt is a noisy version of the input x. The corresponding objective can be simplified to (Sec. A)

Unlike previous work that relied on autoregressive, attention-based transformer models in a highly compressed, discrete latent space [21, 61, 96], we can take advantage of image-specific inductive biases that our model offers.
This includes the ability to build the underlying UNet primar- ily from 2D convolutional layers, and further focusing the objective on the perceptually most relevant bits using the reweighted bound, which now reads
```

- 拡散モデル[77]は、長さTの固定マルコフ連鎖の逆過程を学習することに相当するノルマル分布変数を徐々にノイズ除去することによってデータ分布p(x)を学習するように設計された確率的モデルである。
- 画像合成に関して、最も成功したモデル [14、27、67]は、ノイズ除去スコア・マッチング [80] と同じ$p(x)$の変分低界の再可重変形に依拠している。
- これらのモデルは，入力$x_t$のノイズ除去された変種を予測するように訓練された等重量化オートエンコーダ$\epsilon_\theta (x_t,t); t=1 ... T$のシーケンスとして解釈することができる. 
- ここで$x_t$は入力$x$にノイズを付加したもので、$t$は一連のシーケンスのインデックス
- これに対応する目的は以下のように単純化できる。

$$
\large L_{LDM} = \mathbb{E}_{x, \epsilon \textasciitilde \mathcal{N}(0, 1), t} \
	\lbrack \lVert \epsilon - \epsilon_\theta(x_t, t) \rVert ^2_2 \rbrack
$$

### Geenerative Modeling of Latent Representations

```
With our trained perceptual compression models consisting of $\mathcal{E}$ and $\mathcal{D}$, we now have access to an efficient, low-dimensional latent space in which high-frequency, imperceptible details are abstracted away.
Compared to the high-dimensional pixel space, this space is more suitable for likelihood-based generative models, as they can now (i) focus on the important, semantic bits of the data and (ii) train in a lower dimensional, computationally much more efficient space.

Unlike previous work that relied on autoregressive, attention-based transformer models in a highly compressed, discrete latent space [21, 61, 96], we can take advantage of image-specific inductive biases that our model offers.
```

- $\mathcal E$と$\mathcal{D}$で構成される知覚圧縮モデルの学習により、高周波数で知覚できない細部が抽象化された、効率的で低次元の潜在空間にアクセスすることができるようになる
- この空間は、高次元の画素空間と比較して、尤度ベースの生成モデルに適しており、(i)データの重要な意味的ビットに焦点を当て、(ii)低次元で計算効率の高い空間で学習することができる。
- 高度に圧縮された離散的な潜在空間における自己回帰的な注意に基づく変換モデル[21, 61, 96]に依存した先行研究とは異なり、我々のモデルが提供する画像固有の誘導バイアスを利用することができます。
- これには、主に2次元畳み込み層から基礎となるUNetを構築する機能と、再重み付け境界を使用して知覚的に最も関連性の高いビットに目的を集中させる機能が含まれており、以下の式で表せる。

$$
\large L_{LDM} := \mathbb{E}_{\mathcal{E}(x), \epsilon \textasciitilde \mathcal{N}(0, 1), t} \
	\lbrack \lVert \epsilon - \epsilon_\theta(z_t, t) \rVert ^2_2 \rbrack
$$

```
The neural backbone εθ(◦,t) of our model is realized as a time-conditional UNet [66]. Since the forward process is fixed, zt can be efficiently obtained from E during training, and samples from p(z) can be decoded to image space with a single pass through D.
```

- 本モデルのニューラルバックボーン$\epsilon_\theta(\circ, t)$は、時間条件付きUNet[66]として実現されている。
- フォワードプロセスが固定されているため、$z_t$は訓練中に$ \mathcal{E} $から効率的に得ることができ、$p(z)$からのサンプルは$\mathcal{D}$を1回通過するだけで画像空間へ復号することができる。

### 3.3 Conditioning Mechanisms

```
Similar to other types of generative models [51, 78], diffusion models are in principle capable of modeling conditional distributions of the form p(z|y).

This can be implemented with a conditional denoising autoencoder εθ(zt, t, y) and paves the way to controlling the synthesis process through inputs y such as text [63], semantic maps [29, 55] or other image-to-image translation tasks [30].

In the context of image synthesis, however, combining the generative power of DMs with other types of condition- ings beyond class-labels [14] or blurred variants of the input image [67] is so far an under-explored area of research.

We turn DMs into more flexible conditional image generators by augmenting their underlying UNet backbone with the cross-attention mechanism [91], which is effective for learning attention-based models of various input modalities [31,32].
To pre-process y from various modalities (such as language prompts) we introduce a domain specific encoder $\tau_\theta$ that projects $y$ to an intermediate representation τθ(y) ∈ RM×dτ , which is then mapped to the intermediate layers of the UNet via a cross-attention layer implementing Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt d}) \cdot V$
```

- 他のタイプの生成モデル[51, 78]と同様に、拡散モデルは原理的にp(z|y)の形の条件付き分布をモデル化することが可能である。

- これは条件付きノイズ除去オートエンコーダ$\epsilon_\theta(z_t, t, y)$で実装でき、テキスト[63]、意味マップ[29, 55]、その他の画像間翻訳タスク[30]などの入力$y$を通して合成プロセスを制御する道を開くものである。

- しかし、画像合成の文脈では、クラスラベル[14]や入力画像の不鮮明なバリエーション[67]以外の他のタイプの条件とDMの生成力を組み合わせることは、今のところ未開拓の研究分野である。

- 我々は、DMをより柔軟な条件付き画像生成器とするために、その基礎となるUNetバックボーンを、様々な入力モダリティのAttentionベースのモデルの学習に有効なCross-Attentionメカニズム[91]で補強する[31、32]。

- 様々なモダリティ（言語プロンプトなど）からの$y$を前処理するために、ドメイン固有のエンコーダー$\tau_\theta$を導入する

  - これは, $y$を中間表現$\tau_\theta(y) \in R^M \times d_\tau$に投影して、$Attention(Q, K, V) = softmax(\frac{QK^T}{sqrt d}) \cdot V $が実装された, Cross-Attentionレイヤを介して, UNetの中間層にマッピングされる

    
    $$
    Q = W^{(i)}_{Q} . \mathcal{p}_i(z_t), \\
    K= W^{(i)}_{K} . \tau_\theta(y), \\
    V = W^{(i)}_{V} . \tau_\theta(y) \\
    $$
    
  - 

  ```
  Here, $ \varphi_i(z_t) \in \mathbb{R}^{N \times d^i_\epsilon} $ denotes a (flattened) intermediate
  representation of the UNet implementing and $W^{(i)}_V, W^{(i)}_Q, W^{(i)}_{K}$ are learnable projection matrices [32, 91]. See Fig. 3 for a visual depiction.
  
  Based on image-conditioning pairs, we then learn the conditional LDM via
  $$
  \large L_{LDM} := \mathbb{E}_{x, y, \epsilon \textasciitilde \mathcal{N}(0, 1), t} \
  	\lbrack \lVert \epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y)) \rVert ^2_2 \rbrack
  $$
  
  where both tau_theta and epsilon_theta are jointly optimized via Eq. 3.
  This conditioning mechanism is flexible as tau_theta can be parameterized with domain-specific experts, e.g. (unmasked) transformers [91] when y are text prompts (see Sec. 4.3.1)
  ```

  

- ここで$ \varphi_i(z_t) \in \mathbb{R}^{N \times d^i_\epsilon} \\ $はUNetの平坦化された中間表現

- $W^{(i)}_V, W^{(i)}_Q, W^{(i)}_{K}$ は学習可能な射影行列を表す。

- 画像-条件ペアに基づき、条件付きLDMを学習することは以下のように定式化できる。

$$
\large L_{LDM} := \mathbb{E}_{x, y, \epsilon \textasciitilde \mathcal{N}(0, 1), t} \
	\lbrack \lVert \epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y)) \rVert ^2_2 \rbrack
$$

- ここで、$\tau_θ$と$\epsilon_θ$の両方は、式4によって共同で最適化される。
- この条件付けのメカニズムは柔軟であり、$\tau_θ$はドメイン固有のエキスパート、例えば$y$がテキストプロンプトの場合は（マスクされていない）変換器[91]でパラメータ化できる（4.3.1項参照）。

## 4. Experiments

```
LDMs provide means to flexible and computationally tractable diffusion based image synthesis also including
high-resolution generation of various image modalities, which we empirically show in the following. Firstly, however, we analyze the gains of our models compared to pixelbased diffusion models in both training and inference.
Interestingly, we find that LDMs trained in VQ-regularized latent spaces achieve better sample quality, even though the reconstruction capabilities of VQ-regularized first stage models slightly fall behind those of their continuous counterparts, cf . Tab. 7.

Therefore, we evaluate VQ-regularized LDMs in the remainder of the paper, unless stated differently.
A visual comparison between the effects of first stage regularization schemes on LDM training and their generalization abilities to resolutions higher than 2562 can be found in Appendix C.1.
In D.2 we furthermore list details on architecture, implementation, training and evaluation for all results presented in this section.
```

- LDMは、様々な画像モダリティの高解像度生成を含む、柔軟で計算しやすい拡散ベースの画像合成の手段を提供し、我々はそれを以下に実証的に示すものである。
- まず、学習と推論の両方において、ピクセルベースの拡散モデルと比較して、我々のモデルの利点を分析する。
- 興味深いことに、VQ正則化された潜在空間で学習したLDMは、VQ正則化された初段モデルの再構成能力が連続モデルの再構成能力にわずかに及ばないにもかかわらず、より優れたサンプル品質を達成することがわかった（cf . 表7を参照)
- したがって、本論文の残りの部分では、特に断りのない限り、VQ正則化LDMを評価する。
- LDMのトレーニングにおける第一段階の正則化スキームの効果と、2562より高い解像度への一般化能力の視覚的な比較は、付録C.1に記載されている。
- D.2では、さらに、このセクションで紹介するすべての結果について、アーキテクチャ、実装、トレーニング、評価の詳細を列挙する。 

### 4.1. Latent Diffusion Models are Efficient and Faithful Image Generators

```
This section analyzes the behavior of our LDMs with dif- ferent downsampling factors f ∈ {1, 2, 4, 8, 16, 32} (abbreviated as LDM-f , where LDM-1 corresponds to pixel-based DMs). 
To obtain a comparable test-field, we fix the computational resources to a single NVIDIA A100 for all ex- periments in this section and train all models for the same number of steps and with the same number of parameters.
```

- 本節では、異なるダウンサンプリング係数 f∈{1, 2, 4, 8, 16, 32}（LDM-fと略記、LDM-1はピクセルベースのDMに対応）を持つ LDMの挙動を分析する。
- 本節では、比較可能なテストフィールドを得るために、計算機資源を1台のNVIDIA A100に固定し、すべてのモデルを同じステップ数、同じパラメータ数で学習させる実験を行った。

```
Tab. 1 shows hyperparameters and reconstruction performance of the first stage models used for the LDMs compared in this section. 

Fig. 5 shows sample quality as a function of training progress for 2M steps of class conditional models on the ImageNet [11] dataset. 
We see that, i) small downsampling factors for LDM-{1,2} result in slow train- ing progress, whereas ii) overly large values of f cause stag- nating fidelity after comparably few training steps.

Revisiting the analysis above (Fig. 1 and 2) we attribute this to 
i) leaving most of perceptual compression to the diffusion model and 
ii) too strong first stage compression resulting in information loss and thus limiting the achievable quality. LDM-{4-16} strike a good balance between efficiency and perceptually faithful results, which manifests in a significant FID [26] gap of 38 between pixel-based diffusion (LDM-1) and LDM-8 after 2M training steps.
```

- 表1には使用したハイパラと第一段階での画像再構成のパフォーマンスを比較が示されている。
- 図5は、ImageNet [11]データセットにおいて、クラス条件モデルを2Mステップで学習させた場合の、学習進度の関数としてのサンプル品質である。
  **LDM-{1,2}のダウンサンプリング係数が小さいと学習の進捗が遅くなり、fの値が大きすぎると、比較的少ない学習ステップで忠実度が低下することがわかる。　-> f=4-16が最適らしい**
- 上記の分析（図1、図2）を再確認すると、次のような理由があることがわかります。
  i) 知覚的な圧縮の大部分を拡散モデルに委ねたこと。
  ii)第一段階の圧縮が強すぎて情報が失われ、達成可能な品質が制限される。LDM-{4-16}は、効率と知覚に忠実な結果の間で良いバランスを取っており、これは2Mトレーニングステップ後のピクセルベース拡散（LDM-1）とLDM-8の間の38という有意なFID [26] ギャップで現れています。

```
In Fig. 6, we compare models trained on CelebA-HQ [35] and ImageNet in terms sampling speed for different numbers of denoising steps with the DDIM sampler [79] and plot it against FID-scores [26].
LDM-{4-8} outperform models with unsuitable ratios of perceptual and conceptual compression.
Especially compared to pixel-based LDM-1, they achieve much lower FID scores while simultaneously significantly increasing sample throughput.
Complex datasets such as ImageNet require reduced compression rates to avoid reducing quality.
Summarized, we observe that LDM-4 and -8 lie in the best behaved regime for achieving high-quality synthesis results.
```

図6では、CelebA-HQ [35]とImageNetで学習したモデルを、**DDIMサンプラー [79]を用いて異なるノイズ除去ステップ数のサンプリング速度で比較し、FID-スコア [26]に対してプロットしています。**
LDM-{4-8}は、知覚的圧縮と概念的圧縮の比率が不適当なモデルを凌駕しています。
特にピクセルベースのLDM-1と比較すると、サンプルのスループットを大幅に向上させると同時に、はるかに低いFIDスコアを達成した。ImageNetのような複雑なデータセットでは、品質の低下を避けるために圧縮率を下げる必要がある。
以上のことから、**LDM-4とLDM-8は、高品質な合成結果を得るために最も適した領域であることがわかります。**

![image-20230619220428169](/Users/yuki/Library/Application Support/typora-user-images/image-20230619220428169.png)

- CelebA-HQ と ImageNetで圧縮率の異なるLDMを比較
- 破線は200stepのFIDスコアを示し, 異なる圧縮比のモデルと比較して, LDM4-8の性能が高いことを示している

### 4.2 Image Generation with Latent Diffusion

```
We train unconditional models of 2562 images on CelebA-HQ [35], FFHQ [37], LSUN-Churches and LSUN-Bedrooms [95] and evaluate the i) sample quality and ii) their coverage of the data manifold using ii) FID [26] and ii) Precision-and-Recall [46]. 
Tab. 2 summarizes our results. 

On CelebA-HQ, we report a new state-of-the-art FID of 5.11, outperforming previous likelihood-based models as well as GANs. We also outperform LSGM [87] where a latent diffusion model is trained jointly together with the first stage.

In contrast, we train diffusion models in a fixed space and avoid the difficulty of weighing reconstruction quality against learning the prior over the latent space, see Fig. 1-2, and Tab. 1.

We outperform prior diffusion based approaches on all but the LSUN-Bedrooms dataset, where our score is close to ADM [14], despite utilizing half its parameters and requiring 4-times less train resources (see Appendix D.3.5).

Moreover, LDMs consistently improve upon GAN-based methods in Precision and Recall, thus confirming the advantages of their mode-covering likelihood-based training objective over adversarial approaches.
In Fig. 4 we also show qualitative results on each dataset.
```

CelebA-HQ [35], FFHQ [37], LSUN-Churches and LSUN-Bedrooms [95]の2562画像の無条件モデルを学習し、ii) FID [26] と ii) Precision-and-Recall [46] を用いて i) サンプルの品質と ii) データ多様体のカバー率を評価した。
表2がその結果である。

- **CelebA-HQにおいて、我々は5.11という最新のFIDを報告し、これまでの尤度ベースモデルやGANを凌駕した。**
  また、潜在拡散モデルを第1ステージと共同で学習させるLSGM [87]をも凌駕する。
- 一方、我々は固定空間で拡散モデルを訓練し、潜在空間上の事前学習と再構成の質を比較検討する難しさを回避している（図1-2、およびTab.1参照）。1.
- LSUN-Bedroomsデータセットでは、ADM [14]の半分のパラメータを利用し、4倍の学習リソースを必要とするにもかかわらず、ADM [14]に迫るスコアを記録した。（付録D.3.5参照）我々は拡散ベースのアプローチ以外では、先行アプローチを上回る性能を持つ。]
- さらに、LDMはGANベースの手法よりもPrecisionとRecallで一貫して向上しており、モードカバー尤度ベースの学習目的が敵対的アプローチよりも優れていることが確認されました。
  図4では、各データセットの定性的な結果も示しています。

### 4.3. Conditional Latent Diffusion

#### 4.3.1. Transformer Encoders for LDMs

```
By introducing cross-attention based conditioning into LDMs we open them up for various conditioning modalities previously unexplored for diffusion models. For a text-to- image model, we pre-train models conditioned on language prompts on the LAION [73] database and finetune and eval- uate on Conceptual Captions [74].

We employ the BERT-tokenizer [13] and implement τθ as a transformer [91] to infer a latent code which can be readily employed by our conditional LDM.
While this combination of domain specific experts for learning a language representation and visual synthesis results in quite a powerful model, with significantly reduced parameter count and increased throughput compared to AR methods, we view these results as preliminary and expect them to further improve by (i) scaling the transformer backbone τθ and (ii) increasing quality of textimage pairs and (iii) providing time-conditioning to τθ.

The results can be reviewed in Tab. 3.
Fig. 7 (bottom) depicts qualitative examples of the generalization abilities of our model to user defined language prompts.
To further analyze the flexibility of the cross-attention based conditioning mechanism we also train models to synthesize images based on semantic layouts on OpenImages [45], and finetune on COCO [4], see Fig. 7. See Sec. C.3 for the quantitative evaluation and implementation details.
Lastly, following prior work [3, 14, 19, 21], we evaluate our best-performing class-conditional ImageNet models with f ∈ {4, 8} from Sec. 4.1 in Fig. 4 and Sec. C.5.
```

- LDMにクロスアテンションベースの条件付けを導入することで、これまで拡散モデルで未開拓だった様々な条件付けの様式に対応できるようになります。テキストから画像へのモデルの場合、LAION [73]データベースで言語プロンプトに条件付けしたモデルを事前学習し、Conceptual Captions [74]で微調整と評価を行っている。
- 我々はBERT-tokenizer [13]を採用し、$\tau_\theta$を変換器 [91]として実装し、我々の条件付きLDMが容易に採用できる潜在コードを推論することができる。
- 言語表現と視覚的合成を学習するためのドメイン固有の専門家のこの組み合わせは、AR手法と比較してパラメータ数が大幅に減少し、スループットが向上した、非常に強力なモデルをもたらすが、我々はこれらの結果を予備的とみなし、（i）変換器バックボーン$\tau_\theta$のスケーリング、（ii）テキスト画像ペアの品質向上、（iii）$\tau_\theta$に時間条件を与えることによってさらに向上すると予想している。

- その結果は、表3で確認することができる。
- 図7（下）は、ユーザー定義の言語プロンプトに対する本モデルの汎化能力の定性的な例を示したものである。

- クロスアテンションベースの条件付けメカニズムの柔軟性をさらに分析するために、OpenImages [45]で意味的なレイアウトに基づく画像を合成するモデルを訓練し、COCO [4]で微調整することも行った（図7参照）。定量的な評価と実装の詳細については、セクションC.3を参照されたい。
  最後に、先行研究[3, 14, 19, 21]に従い、4.1節のf∈{4, 8}で最も性能の良いクラス条件付きImageNetモデルの評価を図4とC.5節で行った。

#### 4.3.2 Convolutional Sampling Beyond $256^2$

```
By concatenating spatially aligned conditioning information to the input of εθ, LDMs can serve as efficient general purpose image-to-image translation models.
We use this to train models for semantic synthesis, super-resolution (Sec. 4.4) and inpainting (Sec. 4.5).
For semantic synthesis, we use images of landscapes paired with semantic maps [21, 55] and concatenate downsampled versions of the semantic maps with the latent image representation of a f = 4 model (VQ-reg., see Tab. 1).

We train on an input resolution of 256^2 (crops from 384^2) but find that our model generalizes to larger resolutions and can generate images up to the megapixel regime when evaluated in a convolutional manner (see Fig. 8).

We exploit this behavior to also apply the super-resolution models in Sec. 4.4 and the inpainting models in Sec. 4.5 to generate large images between 512^2 and 1024^2.

For this application, the signal-to-noise ratio (induced by the scale of the latent space) significantly affects the results. 
In Sec. C.1 we illustrate this when learning an LDM on (i) the latent space as provided by a f=4 model (KL-reg., see Tab. 1), and (ii) a rescaled version, scaled by the component-wise standard deviation.
```

- 空間的に整列した条件付け情報をεθの入力に連結することで、LDMは効率的な汎用画像間変換モデルとして機能する。
  これを用いて、意味合成、超解像（第4.4節）、インペインティング（第4.5節）のモデルを学習する。
  - 意味合成では、意味マップ[21, 55]と対になった風景の画像を用い、意味マップのダウンサンプル版とf=4モデルの潜像表現を連結する（VQ-reg.、表1参照）。
- 入力解像度は256^2（384^2からのクロップ）ですが、このモデルはより大きな解像度に一般化し、コンボリューション方式で評価するとメガピクセル領域まで画像を生成できることがわかります（図8参照）。
  - この挙動を利用して、4.4節の超解像モデルと4.5節のインペインティングモデルを適用し、512^2～1024^2の大きな画像を生成することもできます。
- このアプリケーションでは、（潜在空間のスケールによって引き起こされる）信号対雑音比が結果に大きく影響する。
  C.1では、(i)f=4モデル（KL-reg.、表1参照）によって提供される潜在空間と、(ii)成分ごとの標準偏差でスケーリングされた再スケーリングバージョンでLDMを学習したときのことを説明する。

### 4.4 Super-Resolution with Latent Diffusion

```
LDMs can be efficiently trained for super-resolution by diretly conditioning on low-resolution images via concate- nation (cf . Sec. 3.3).

In a first experiment, we follow SR3 [67] and fix the image degradation to a bicubic interpolation with 4×downsampling and train on ImageNet following SR3’s data processing pipeline. 

We use the f=4 autoencoding model pretrained on OpenImages (VQ-reg., cf . Tab. 1) and simply concatenate the low-resolution conditioning y and the inputs to the UNet, i.e. τθ is the identity.
Our qualitative and quantitative results (see Fig. 10 and Tab. 4) show competitive performance and LDM-SR outperforms SR3 in FID while SR3 has a better IS.

A simple image regression model achieves the highest PSNR and SSIM scores; however these metrics do not align well with human perception [99] and favor blurriness over imperfectly aligned high frequency details [67]. 

We can still push these metrics by using a post-hoc guiding mechanism [14] and we implement this image-based guider via a perceptual loss, see Sec. C.7.
```

- LDMは、連結によって低解像度画像に直接条件付けすることで、超解像のための学習を効率的に行うことができる（参照：Sec.3.3）。
- 最初の実験では、SR3 [67]に従い、画像劣化を4×ダウンサンプリングのバイキュービック内挿に固定し、SR3のデータ処理パイプラインに従ってImageNetで学習させることにした。
- OpenImagesで事前学習したf=4自動符号化モデル（VQ-reg., cf . Tab. 1）を用い、低解像度条件付けyとUNetへの入力を単に連結する、すなわちτθを同一とする。
- 定性的・定量的な結果（図10、Tab.4参照）は、競争力のある性能を示し、LDM-SRはFIDでSR3を上回り、SR3はISで優れている。
- 単純な画像回帰モデルは最高のPSNRとSSIMスコアを達成しましたが、これらの指標は人間の知覚[99]とあまり一致せず、不完全に整列した高周波の詳細よりもぼやけた状態を優先します[67]。
- 我々は、ポストホックガイドメカニズム[14]を使用することにより、これらのメトリックを押し上げることができ、我々はこの画像ベースのガイドを知覚的損失によって実装します。

```
To evaluate generalization of our LDM-SR, we apply it both on synthetic LDM samples from a class-conditional ImageNet model (Sec. 4.1) and images crawled from the internet. 

Interestingly, we observe that LDM-SR, trained only with a bicubicly downsampled conditioning as in [67], does not generalize well to images which do not follow this pre-processing.

Hence, to obtain a model for generic images, we train LDM-BSR, which adopts a more diverse degradation process following [98]. 

Fig. 9 illustrates the effective- ness of this approach. LDM-BSR produces images much sharper than the models confined to a fixed pre-processing, making it suitable for real-world applications.
```

- LDM-SRの汎化を評価するため、クラス条件付きImageNetモデル（Sec4.1）から得た合成LDMサンプルとインターネットからクローリングした画像の両方に適用した。
- 興味深いことに、[67]のように2次元的にダウンサンプリングした条件付けのみで学習したLDM-SRは、この前処理を行わない画像にはうまく一般化しないことがわかる。
- そこで、一般的な画像のモデルを得るために、[98]に倣ってより多様な劣化処理を採用したLDM-BSRを学習させる。
- 図9は、このアプローチの有効性を示しています。LDM-BSRは、固定された前処理に限定されたモデルよりもはるかにシャープな画像を生成し、実世界での応用に適しています。

### 4.5 Inpainting with Latent Diffusion

```
Inpainting is the task of filling masked regions of an image with new content either because parts of the image are are corrupted or to replace existing but undesired content within the image.
We evaluate how our general approach for conditional image generation compares to more specialized, state-of-the-art approaches for this task.
Our evaluation follows the protocol of LaMa [83], a recent inpainting model that introduces a specialized architecture relying on Fast Fourier Convolutions [7].
We describe the exact training & evaluation protocol on Places [101] in Sec. D.2.2.

We first analyze the effect of different design choices for the first stage.
We compare the inpainting efficiency of LDM-1 (i.e. a pixel-based conditional DM) with LDM-4, for both KL and VQ regularizations, as well as VQ-LDM-4 without any attention in the first stage (see Tab. 1), where the latter reduces GPU memory for decoding at high resolutions.

For comparability, we fix the number of parameters for all models. 
Tab. 5 reports the training and sampling throughput at resolution 256^2 and 512^2, the total training time in hours per epoch and the FID score on the validation split after six epochs.
Overall, we observe a speed-up of at least 2.7x between pixel- and latent-based diffusion models while improving FID scores by a factor of at least 1.6x.
```

- インペインティングとは、画像の一部が破損しているため、あるいは画像内の既存の望ましくないコンテンツを置き換えるために、画像のマスクされた領域を新しいコンテンツで埋めるタスクである。
- 我々は、条件付き画像生成のための我々の一般的なアプローチが、このタスクのためのより専門的で最先端のアプローチと比較してどうであるかを評価する。
  我々の評価は、高速フーリエ変換[7]に依存する特殊なアーキテクチャを導入した最近のインペインティングモデルであるLaMa[83]のプロトコルに従っています。Places [101]の正確な訓練と評価プロトコルについては、D.2.2節で説明します。

- まず、第一段階での異なる設計選択の効果を分析する。
  LDM-1（すなわちピクセルベースの条件付きDM）のインペイント効率を、KL正則化とVQ正則化の両方についてLDM-4と比較し、さらに第1ステージに注意を払わないVQ-LDM-4（タブ1参照）、後者は高解像度でのデコードのためにGPUメモリを削減します。

- 比較のため、すべてのモデルでパラメータ数を固定しています。
- 表 5では、解像度256^2および512^2でのトレーニングおよびサンプリングのスループット、1エポックあたりのトレーニング時間（時間）、6エポック後の検証分割でのFIDスコアが報告されている。
- **全体として、ピクセルベースと潜在ベースの拡散モデルの間で少なくとも2.7倍のスピードアップが見られ、同時にFIDスコアは少なくとも1.6倍向上していることが確認された。**

```
The comparison with other inpainting approaches in Tab. 6 shows that our model with attention improves the overall image quality as measured by FID over that of [83].
LPIPS between the unmasked images and our synthesized images is slightly higher than that of [83], probably due to the fact that [83] can only produce a single result which tends to recover more of an average image compared to the diverse results produced by our LDM as shown in Fig. 11.

Based on these initial results, we also trained a larger diffusion model (big in Tab. 6) in the latent space of the VQ-regularized first stage without attention.
Following [14], the UNet of this diffusion model uses attention layers on three levels of its feature hierarchy, the BigGAN [3] residual block for up- and downsampling and has 387M parameters instead of 215M.

After training, we noticed a discrepancy in the quality of samples produced at resolutions 256^2 and 512^2, which we hypothesize to be caused by the additional attention modules.
However, fine-tuning the model for half an epoch at resolution 512^2 allows the model to adjust to the new feature statistics and sets a new state of the art FID on image inpainting (big, w/o attn, w/ ft in Tab. 6, Fig. 12.).
```

- 表6では、他のインペインティングアプローチと比較しています。6では、注意を喚起する我々のモデルが、[83]のものよりFIDで測定される全体的な画質を向上させることが示されています。
- マスクなし画像と我々の合成画像の間のLPIPSは、[83]よりもわずかに高いが、これは、[83]が単一の結果しか出せないため、図11に示すように我々のLDMが生み出す多様な結果に比べて、より平均的な画像を回復する傾向があることが原因であると考えられる。
- これらの初期結果に基づいて、我々はまた、より大きな拡散モデル（表6のビッグ）を、注意を伴わないVQ正則化第1ステージの潜在空間で訓練した。
  14]に従い、この拡散モデルのUNetは、特徴階層の3つのレベルにAttn層を用い、アップ・ダウンサンプリングにBigGAN[3]残差ブロックを用い、215Mの代わりに387Mのパラメータを持つ。
- 学習後、解像度256^2と512^2で生成されたサンプルの品質に矛盾があることに気づいたが、これは注意モジュールが追加されたことに起因しているという仮説を立てた。
- しかし、解像度512^2で半エポックの間モデルを微調整することで、モデルは新しい特徴統計に適応し、画像インペインティングに関する新しい状態のFIDを設定しました（Tab.6、図12.の大きな、w/o attn、w/ ft）。

## 5. Limitations & Sociental Impact

### Limitations

```
While LDMs significantly reduce computational requirements compared to pixel-based approaches, their sequential sampling process is still slower than that of GANs.
Moreover, the use of LDMs can be questionable when high precision is required: although the loss of image quality is very small in our f=4 autoencoding models (see Fig. 1), their reconstruction capability can become a bottleneck for tasks that require fine-grained accuracy in pixel space.
We assume that our superresolution models (Sec. 4.4) are already somewhat limited in this respect.
```

- LDMは既存モデルよりも大幅に計算量を削減できた一方, 逐次的サンプリング処理がGANよりも遅い
- 高精度が必要とされる場合はLDMの使用に疑問が残る。
  - f=4の自動符号化モデルでは、画像品質の低下が小さい
  - ピクセル空間での微細な精度が求められるタスクでは、細部まで再現できないため不利である。

### Societal Impact

```
Generative models for media like imagery are a double-edged sword: On the one hand, they enable various creative applications, and in particular approaches like ours that reduce the cost of training and inference have the potential to facilitate access to this technology and democratize its exploration.
On the other hand, it also means that it becomes easier to create and disseminate manipulated data or spread misinformation and spam.
In particular, the deliberate manipulation of images (“deep fakes”) is a common problem in this context, and women in particular are disproportionately affected by it [12, 22].

Moreover, deep learning modules tend to reproduce or exacerbate biases that are already present in the data [20,34,85].
While diffusion models achieve better coverage of the data distribution than e.g. GAN-based approaches, the extent to which our two-stage approach that combines adversarial training and a likelihood-based objective misrepresents the data remains an important research question.
For a more detailed discussion of the ethical considera- tions of deep generative models, see e.g. [12].
```

画像のようなメディアの生成モデルは、諸刃の剣である： 特に、学習と推論のコストを削減する我々のアプローチは、この技術へのアクセスを容易にし、その探求を民主化する可能性を持っています。
その一方で、操作されたデータを作成し広めることや、誤った情報やスパムを広めることが容易になることも意味しています。
特に、画像の意図的な操作（「ディープフェイク」）はこの文脈でよく見られる問題であり、特に女性はその影響を不当に受けています[12, 22]。

さらに、深層学習モジュールは、データに既に存在するバイアスを再現したり悪化させたりする傾向がある[20,34,85]。
拡散モデルはGANベースのアプローチなどよりもデータ分布の良好なカバレッジを達成するが、敵対的な訓練と尤度ベースの目的を組み合わせた我々の2段階のアプローチがどの程度データを誤って表現するかは、依然として重要な研究課題である。

深層生成モデルの倫理的考察については、例えば[12]を参照されたい。

## 6. Conclusion

```
We have presented latent diffusion models, a simple and efficient way to significantly improve both the training and sampling efficiency of denoising diffusion models without degrading their quality. Based on this and our cross-attention conditioning mechanism, our experimemnnts could demonstrate favorable results compared to state-of-the-art methods across a wide range of conditional image synthesis tasks without task-specific architecures.
```

- 我々は、潜在拡散モデルを提示し、その品質を低下させることなく、ノイズ除去拡散モデルの学習効率とサンプリング効率を大幅に改善する簡単で効率的な方法を示した。
- この方法と、我々のクロスアテンションを用いた条件付け器機構に基づき、我々の実験は、タスクに特化したアーキテクチャを用いずに、広範囲の条件付き画像合成タスクにおいて、最新手法と比較して良好な結果を示すことができた。
