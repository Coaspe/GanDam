Konkuk univ. Spring 2022 Open Source Software Project team3
   
# StarGAN with CBAM(attention module): Generate diabetic-retinopathy images dataset.
   

## Dataset: 35,162 pairs
https://www.kaggle.com/c/diabetic-retinopathy-detection

   
## Proposal presentaion(KR)
https://youtu.be/7j02KCI767A

   
## Datast Labels
#### Left   
0 - No DR   
1 - Mild   
2 - Moderate   
3 - Severe   
4 - Proliferative DR
   

#### Right   
5 - No DR   
6 - Mild   
7 - Moderate   
8 - Severe   
9 - Proliferative DR   
   

## Structure

![image](https://user-images.githubusercontent.com/76432686/170833643-6aa389d6-c426-49c0-a3a8-e41da5a7b614.png)



## 2022-05-28   
Only Generator + CBAM    
Batch size: 16,    
Iteration: about 60000,    
result: fail

![image](https://user-images.githubusercontent.com/76432686/170833551-607ec89b-db5e-4302-89f9-f77e1cf046b5.png)
