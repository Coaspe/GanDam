Konkuk univ. Spring 2022 Open Source Software Project team3
   
# StarGAN with CBAM(attention module): Generate diabetic-retinopathy images dataset.
   
## GPU: NVIDIA Titan X

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

## 2022-05-29
No CBAM.  
Batch size: 16,   
Iteration: about 128,000.  
result: fail.  
![image](https://user-images.githubusercontent.com/76432686/170960623-f0ab19fa-a84e-4459-98ea-87f9e94f931e.png)

## 2022-05-30
Both with CBAM.  
Batch size: 16.  
Iteration: about 60,000.  
result: fail.  
News: We change Generator(CBAM part) and recognize there is large difference in the number of data.   
#### Left 0 ~ 4, Right 0 ~ 4.  
[12871, 1212, 2702, 425, 353, 12939, 1231, 2590, 448, 355]
![image](https://user-images.githubusercontent.com/76432686/170960836-d812ed2e-3f45-4997-89c5-1f7837f1a50f.png)
