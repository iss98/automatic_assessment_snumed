# (2023 SNUMED) Mathematics education assessment class assignment

## Group Memebers
김명식, 김재훈, 김지영, 신인섭, 윤예린, 정유진

## Dataset

Data on 20 problems related to linear equation solving, polynomial and uniaxial operations were collected from second-year middle school students.

## Models
Tokenizer : Bertwordpiece tokenizer and Sentencepiece tokenizer
Models : RNN, LSTM, MultiheadAttention

## Results

|  | **1-1** | **1-2** | **1-3** | **1-4** | **1-5** | **1-6** | **1-7** | **1-8** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RNN | **0.956** | 0.921 | 0.851 | 0.844 | 0.900 | **0.979** | 0.731 | 0.586 |
| LSTM | 0.914 | **0.972**  | 0.924 | 0.816 | **0.950** | 0.853 | **0.766** | **0.744**|
| Attention | 0.979 | 0.921 | **0.879** | **0.849** | 0.944 | 0.876 | 0.670 | 0.706 |

|  | **2-1** | **2-2** | **2-3** | **2-4** | **2-5** | **2-6** | **2-7** | **2-8** | **2-9**|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RNN | 0.957 | 0.945 | 0.896 | 0.896 | **0.861** | 0.628 | 0.908 | 0.741 | 0.797 | 
| LSTM | **0.978** | **0.948** | **0.958** | 0.881 | 0.810 | 0.706 | 0.888 | **0.817** | 0.834 |
| Attention | 0.948 | 0.850 | 0.884 | **0.912** | 0.840 | **0.712** | **0.938** | 0.769 | **0.843**|

|  | **3-1** | **3-2** | **3-3** |
| --- | --- | --- | --- |
| RNN | 0.948 | 0.958 | **0.895** |
| LSTM | **1.000** | **0.969** | 0.854 |
| Attention | 0.938 | 0.875 | 0.860 |