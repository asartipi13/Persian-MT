
# Persian-OtherLanguges  Machine Translation statistics

>In this repo we provide comperhensive statistics for exsiting Machine Translation dataset in Persian. At this stage we just represent statistics Persian-English datasets.

> We also prove Word-Level and Charachter-Level bar chart for each dataset which can be found in folderss dataset.

<br/><br/>
# Persian-English

<p align="center">
Word level <b>Persian-English</b>
</p>

|       dataset       | avg_fa | min_fa | max_fa | 92%_fa |  all_fa | unique_fa | avg_en | min_en | max_en | 92%_en |  all_en | unique_en |
|:-------------------:|:------:|:------:|:------:|:------:|:-------:|:---------:|:------:|:------:|:------:|:------:|:-------:|:---------:|
|  PEPC_Bidirectional |   21   |    6   |   194  |   36   | 4295971 |   140512  |   21   |    7   |   153  |   37   | 4397290 |   134726  |
| PEPC_Onedirectional |   23   |    6   |   194  |   38   | 3650404 |   132567  |   21   |    7   |   153  |   36   | 3391029 |   131110  |
|         TEP         |    7   |    1   |   32   |   14   |  682786 |   36793   |    8   |    1   |   37   |   14   |  716605 |   22615   |

<br/><br/>

<p align="center">
Charachter level <b>Persian-English</b>
</p>

|       dataset       | avgc_fa | minc_fa | maxc_fa | avgc_en | minc_en | maxc_en |
|:-------------------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|  PEPC_Bidirectional |   105   |    23   |   870   |   129   |    17   |   870   |
| PEPC_Onedirectional |   112   |    23   |   870   |   126   |    17   |   870   |
|         TEP         |    33   |    1    |   144   |    37   |    3    |   158   |

![](https://github.com/asartipi13/Persian-MT/blob/main/data/PEPC_Bidirectional/eda/en_ch_length_distrobution.png?raw=true)



Persian word distrobution             |  English word distrobution
:-------------------------:|:-------------------------:
![](https://github.com/asartipi13/Persian-MT/blob/main/data/PEPC_Bidirectional/eda/fa_length_distrobution.png?raw=true)  |  ![](https://github.com/asartipi13/Persian-MT/blob/main/data/PEPC_Bidirectional/eda/en_length_distrobution.png?raw=true)


Persian charachter distrobution             |  English charachter distrobution
:-------------------------:|:-------------------------:
![](https://github.com/asartipi13/Persian-MT/blob/main/data/PEPC_Bidirectional/eda/fa_ch_length_distrobution.png?raw=true)  |  ![](https://github.com/asartipi13/Persian-MT/blob/main/data/PEPC_Bidirectional/eda/en_ch_length_distrobution.png?raw=true)