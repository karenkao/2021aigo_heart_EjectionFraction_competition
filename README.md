# 2021aigo_heart_EjectionFraction_competition
藉由深度學習自動偵測左心室攝影之心臟射出率—建立心衰竭及結構異常之預警平台
## 目地
開發一個心導管左心室攝影心臟分割模型(segmentation)，並計算出左心室射出率（LVEF）之指數。
## 方法
- 心臟顯影劑物件偵測模型: 使用yolov5與YOLOv5l6預訓練權重，偵測該張左心室攝影影像其心臟是否顯影劑充滿，訓練時使用CLAHE提高影像對比與資料增強(Data augmentation)，如mosaic、mixup、剛性變換與非剛性變換，如彈性變換(ElasticTransform)、網格變換(GridDistortion)、光學畸變(OpticalDistortion)、仿射(Affine)等。
- 心臟影像分割模型：使用2.5D unet搭配VGG16為骨幹並採用Imagenet為預訓練權重。訓練時使用CLAHE提高影像對比與14倍資料增強(Data augmentation)包含剛性變換與非剛性變換等。針對損失函數採取二階段訓練，先以combo loss(CE loss, Dice loss) 為初期訓練，待收斂後以focal loss 進行訓練，以解決資料不平衡(imbalance)的問題。
- 計算心臟長軸演算法
  - 最小外接長方形法(下圖左):使用opencv minAreaRect找出該心臟分割mask最小外接方框，其長邊為長軸。
  - 極值點法（下圖中):使用opencv findContours找出該心臟分割mask極值點，即為最左至最右點連線為長軸。
  - 橢圓形法 (下圖右):使用opencv fitEllipse逼進該心臟分割mask之橢圓形 ，其橢圓長軸為長軸。
 
![image](https://user-images.githubusercontent.com/44295049/137444917-e312fad1-011c-4f32-a7d1-53096459ff32.png)

- 計算左心室射出率(LVEF)
  - V=volume, A=心臟面積; L=左心室長軸
  - LVEF = (心臟舒張容積-心臟收縮容積)/心臟舒張容積 * 100 
  
![image](https://user-images.githubusercontent.com/44295049/137442560-3a01f7bb-6c91-41f1-9899-d991fe539afc.png)

## 模型成效
- 心臟顯影劑物件偵測模型(僅training validation):
- 心臟影像分割模型(測試於14個左心室攝影)：
- 計算心臟長軸演算法(測試於14個左心室攝影):
