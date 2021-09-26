# human_segmentation
### 概要
- [人間タワーバトル](https://github.com/ak0592/unilab-tower-battle) におけるリアルタイムでカメラの画像から人を切り出すためのリポジトリ
- 学習済みのモデルとして [u2net_human_seg](https://github.com/axinc-ai/ailia-models/tree/master/image_segmentation/u2net-human-seg) を用いている  

### 初めに
- 学習済みモデルを使う際に、学習済みモデルがまとめられたModuleである `ailia`を手動でインストールする必要がある  
- [このサイト](https://medium.com/axinc/ailia-sdk-チュートリアル-python-28379dbc9649) に手順が書いてある（特にライセンスの置く位置に注意）
### インストール
```shell
git clone git@github.com:ak0592/human_segmentation.git
```
### 使い方
1. プロセスの実行  
```shell
 python real_time_inference.py
```
　　をすると、数行のlogと共にパソコンのインカメからの映像を映したwindowが出てくる  

2. windowを選択して、切り取りたいタイミングでSpaceキーを押す  
(画像のキャプチャ→切り出しが実行され、UnityのSourceフォルダに保存される)
3. windowを選択して、Enterキーを押すと終了

### その他
リアルタイム以外でも、`inference.py`を使うと１枚の画像の切り出しをすることができる
1. `images/source_images`に切り出したい画像を追加
2. プロセスの実行
```shell
     python inference.py <画像ファイル名>.png
```
　　をすると、数行のlogと共に`images/result_images`に切り出された画像が保存される
- 実行例が `images`の中に入っている
### Reference
- [u2net document](https://github.com/NathanUA/U-2-Net)
- [human segmentation dataset](https://github.com/VikramShenoy97/Human-Segmentation-Dataset)
