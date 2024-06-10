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
