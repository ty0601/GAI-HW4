# GAI-HW4

## Theoretical Justification
實作Example 1 整合了DDPM和DIP的方法，利用兩種方法的優勢來進行圖像生成任務。
#### DDPM和DIP的優勢
DDPM：一種生成模型，通過擴散過程產生高質量的圖像，逐步從隨機噪聲去噪成結構化圖像。DDPM在生成逼真圖像方面表現出色，但可能**計算量大且收斂速度慢**。
DIP：利用卷積神經網絡的implicit regularization properties來恢復圖像，而**不需要大量數據集，且細節和紋理方面非常有效**。

因此這次實驗希望利用pretrained DIP model來當作DDPM 中denoise 的initial prior，來改善DDPM 計算量大且收斂速度慢的缺點。
預期使用DIP model所訓練出來的ddpm model，fid score會比沒有結合dip model下降的更快。

#### Potential benefits
從DIP初始化可以減少DDPM生成高質量樣本所需的擴散步驟數，從而加快訓練和推理速度以及收斂速度。
#### Potential limitations
雖然可能減少總的擴散步驟數，但初始的DIP訓練階段增加了整體計算成本。
同時也會依賴DIP性能：最終圖像的質量部分依賴於DIP模型的性能。如果DIP的initial prior 效果不如預期，其帶來的好處可能會減少。所以這次也會討論的不同架構的DIP model作為。

## Experimental Verification

### DIP model
首先先訓練兩個DIP model，一個是用簡單的兩層conv layer，另一個是基於restnet18的架構，並相同訓練30個epoch
#### 2Conv DIP model
![dip_training_loss](https://hackmd.io/_uploads/BJM78EEHC.png)

#### RestNet18-based DIP model 
![dip_RestNet18_training_loss](https://hackmd.io/_uploads/By_7IVNrC.png)

### DDPM
#### DDPM with 2Conv DIP 
##### Loss
![ddpm_dip_training_loss](https://hackmd.io/_uploads/H1yCYVES0.jpg)

#### FID score
![ddpm_dip_fid_scores](https://hackmd.io/_uploads/SycS3V4BA.jpg)

#### Visualization
![image](https://hackmd.io/_uploads/rkElKr4rA.png)

#### DDPM with RestNet18-based DIP model 
##### Loss
![ddpm_dip_RestNet18_training_loss](https://hackmd.io/_uploads/Sksgv44rC.png)

#### FID score
![ddpm_dip_RestNet18_fid_scores](https://hackmd.io/_uploads/rJJd2N4H0.png)

#### Visualization
![image](https://hackmd.io/_uploads/HyHQYrVBR.png)


#### Analysis
在實驗過程中總共使用30000個step，並在每3000 step的時候去算fid score，但因為計算成本很高所以只生成了30個sample去做計算。
從前面兩個fid/step的曲線圖可以看到在第一次算fid score的時候都是最低的，這應該是initi prior的功勞。但是在後面的step中fid score 都有往上的趨勢，以下是我想到的幾個原因:
1. 初始FID score較低，代表說DIP模型作為init prior有效。但隨著訓練進行，DDPM可能逐漸偏離init prior，導致FID score上升。
2. 在融合過程中沒有針對兩個模型的協同訓練進行調整，可能會導致DDPM偏離init prior。

不同架構的DIP model也會影響到DDPM最前面訓練的FID score，在RestNet架構的DIP+DDPM FID 最初的 FID 可以到 140 以下，使這次實驗方法最低的。
未來可能需要更加探討複雜的架構所帶這不到5的fid score下降使否值得。

## Ablation Studies
### Basic DDPM
##### Loss
![ddpm_training_loss](https://hackmd.io/_uploads/Bk5KVHVH0.jpg)

#### FID score
![ddpm_fid_scores](https://hackmd.io/_uploads/SkRLNS4SR.jpg)

#### Visualization
![image](https://hackmd.io/_uploads/B13puHVHA.png)



#### Analysis
原始的DDPM第一次記錄到的FID score都比前面兩個加入DIP model都來的高，更加說明了DIP的效果，但同樣也有越後面的step越高的趨勢，這樣的話有可能是因為FID Score所採的sample數量不夠或是training step不夠導致沒辦法往下收斂。


### Conclusions
1. 加了DIP後對於DDPM前面的training step有明顯的效果。
2. 若要使用需要探討如何更好的融合這兩個方法，史training step中期的不會趨勢向上。
3. 不同的DIP架構會影響DDPM的表現，但用2層conv layer和restnet18的表現差不到5的fid score，需要在探討是否真的帶來效益。


### Implementation
可以到[https://github.com/ty0601/GAI-HW4.git](https://)找到程式碼

##### 先創一個 conda 環境並下載必要套件
```
pip install -r requirements.txt
```
##### 訓練Basic DDPM
```
python DDPM.py
```
##### 訓練DDPM with 2conv DIP
1. Train DIP model
```
python DIP.py
```
2. Train DDPM
```
python DDPM_with_DIP.py
```
##### 訓練DDPM with RestNet18-based DIP
1. Train DIP model
```
python DIP_ResNet18.py
```
2. Train DDPM
```
python DDPM_with_DIP_RestNet18.py
```

##### Inference
結果會存到output file
```
python Inference.py
```
